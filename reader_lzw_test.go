// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"bytes"
	"encoding/binary"
	"image"
	"testing"
)

const (
	lzwClearCode = 256
	lzwEOFCode   = 257
	lzwFirstCode = 258
	lzwMinWidth  = 9
	lzwMaxWidth  = 12
	lzwDictSize  = 1 << lzwMaxWidth
)

// lzwEncoder writes MSB-first TIFF LZW codes, widening one code early as the
// TIFF variant of the format requires.
type lzwEncoder struct {
	buf   bytes.Buffer
	acc   uint32
	bits  uint
	next  int
	width uint
}

func newLZWEncoder() *lzwEncoder {
	e := &lzwEncoder{next: lzwFirstCode, width: lzwMinWidth}
	e.put(lzwClearCode)
	return e
}

// put writes a code without advancing the dictionary; clear and EOI define no entry.
func (e *lzwEncoder) put(code int) {
	e.acc = e.acc<<e.width | uint32(code)
	e.bits += e.width
	for e.bits >= 8 {
		e.bits -= 8
		e.buf.WriteByte(byte(e.acc >> e.bits))
	}
}

// emit writes a data code and advances the dictionary the way a decoder will,
// so the encoder and decoder agree on when the code width grows.
func (e *lzwEncoder) emit(code int) {
	e.put(code)
	if e.next < lzwDictSize {
		e.next++
	}
	if e.next+1 >= 1<<e.width && e.width < lzwMaxWidth {
		e.width++
	}
}

func (e *lzwEncoder) bytes() []byte {
	if e.bits > 0 {
		e.buf.WriteByte(byte(e.acc << (8 - e.bits)))
		e.bits = 0
	}
	return e.buf.Bytes()
}

// lzwTrailingStream emits literals covering the whole strip, then whatever
// trailing writes into the stream: nothing, or a code no dictionary entry
// covers. A reader that stops at the last pixel never sees either.
func lzwTrailingStream(payload int, trailing func(*lzwEncoder)) []byte {
	e := newLZWEncoder()
	e.put(0)
	for i := 1; i < payload; i++ {
		e.emit(i % 256)
	}
	trailing(e)
	return e.bytes()
}

// buildSingleStripLZWTIFF wraps an LZW stream as a one-strip 8-bit grayscale TIFF.
func buildSingleStripLZWTIFF(tb testing.TB, w, h int, stream []byte) []byte {
	tb.Helper()

	const tagCount = 9
	const ifdSize = 2 + tagCount*12 + 4 // count + entries + next-IFD pointer

	buf := binary.LittleEndian.AppendUint16([]byte("II"), 42) // header: "II" + magic 42
	buf = binary.LittleEndian.AppendUint32(buf, 8)            // the IFD leads, its strip follows
	buf = binary.LittleEndian.AppendUint16(buf, tagCount)
	buf = appendIFDEntry(buf, tImageWidth, dtLong, w)
	buf = appendIFDEntry(buf, tImageLength, dtLong, h)
	buf = appendIFDEntry(buf, tBitsPerSample, dtShort, 8)
	buf = appendIFDEntry(buf, tCompression, dtShort, cLZW)
	buf = appendIFDEntry(buf, tPhotometricInterpretation, dtShort, pBlackIsZero)
	buf = appendIFDEntry(buf, tStripOffsets, dtLong, 8+ifdSize)
	buf = appendIFDEntry(buf, tSamplesPerPixel, dtShort, 1)
	buf = appendIFDEntry(buf, tRowsPerStrip, dtLong, h) // one strip
	buf = appendIFDEntry(buf, tStripByteCounts, dtLong, len(stream))
	buf = binary.LittleEndian.AppendUint32(buf, 0) // no next IFD
	return append(buf, stream...)
}

// appendIFDEntry appends one 12-byte TIFF IFD entry with an inline scalar value
// (count 1). A SHORT value occupies the low bytes of the 4-byte value field.
func appendIFDEntry(buf []byte, tag, typ uint16, value int) []byte {
	buf = binary.LittleEndian.AppendUint16(buf, tag)
	buf = binary.LittleEndian.AppendUint16(buf, typ)
	buf = binary.LittleEndian.AppendUint32(buf, 1) // count
	return binary.LittleEndian.AppendUint32(buf, uint32(value))
}

// A strip can hold every pixel it declares and still stop without an EOI code.
// Reading until the decompressor errors discarded a complete strip.
func TestDecodeLZWStripWithoutEOICode(t *testing.T) {
	const w, h = 40, 30

	b := buildSingleStripLZWTIFF(t, w, h, lzwTrailingStream(w*h, func(*lzwEncoder) {}))
	img, err := Decode(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if got, want := img.Bounds(), image.Rect(0, 0, w, h); got != want {
		t.Errorf("bounds = %v, want %v", got, want)
	}
}

// Other encoders leave a code the dictionary never defined after the last
// pixel, which the same over-read surfaced as "lzw: invalid code".
func TestDecodeLZWStripWithTrailingUndefinedCode(t *testing.T) {
	const w, h = 40, 30

	stream := lzwTrailingStream(w*h, func(e *lzwEncoder) { e.put(lzwDictSize - 1) })
	img, err := Decode(bytes.NewReader(buildSingleStripLZWTIFF(t, w, h, stream)))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if got, want := img.Bounds(), image.Rect(0, 0, w, h); got != want {
		t.Errorf("bounds = %v, want %v", got, want)
	}
}

// fillToSlot emits literals until the next free dictionary slot reaches want,
// appending each decoded byte to out. The first code after a clear defines no
// entry, so it is emitted without advancing.
func (e *lzwEncoder) fillToSlot(want int, out []byte) []byte {
	e.put(0)
	out = append(out, 0)
	for i := 1; e.next < want; i++ {
		e.emit(i % 256)
		out = append(out, byte(i%256))
	}
	return out
}

// lzwTableFullStream emits codes until the next free dictionary slot is the last
// one, then uses that code itself, so the decoder must expand it from a live
// prior string. The stream is padded to cover exactly one strip.
func lzwTableFullStream(tb testing.TB, payload int) []byte {
	tb.Helper()

	e := newLZWEncoder()
	written := len(e.fillToSlot(lzwDictSize-2, nil))

	// A two-byte prior string makes the expansion three bytes, so a
	// mis-expansion is short rather than coincidentally the right length.
	e.emit(lzwFirstCode)
	written += 2
	e.emit(lzwDictSize - 1)
	written += 3

	for i := 0; written < payload; i++ {
		e.emit(i % 256)
		written++
	}
	if written != payload {
		tb.Fatalf("stream decodes to %d bytes, want exactly one strip of %d", written, payload)
	}

	e.put(lzwEOFCode)
	return e.bytes()
}

// A strip whose dictionary fills and then uses the code equal to the next free
// slot must still decode. Signalling a full dictionary by clearing last made
// that code expand from an entry that was never defined.
func TestDecodeLZWTableFullStrip(t *testing.T) {
	const w, h = 62, 62

	b := buildSingleStripLZWTIFF(t, w, h, lzwTableFullStream(t, w*h))
	img, err := Decode(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if got, want := img.Bounds(), image.Rect(0, 0, w, h); got != want {
		t.Errorf("bounds = %v, want %v", got, want)
	}
}
