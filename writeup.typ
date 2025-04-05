#import "@preview/ilm:1.4.1": *

#set text(lang: "en")

#show: ilm.with(
  title: [CS 336: Assignment 1],
  author: "Brandon Snider",
  date: datetime(year: 2025, month: 04, day: 15),
  figure-index: (enabled: true),
  table-index: (enabled: true),
  listing-index: (enabled: true),
)

#set enum(numbering: "a)")
#set heading(numbering: none)

= BPE Tokenizer

== Problem (`unicode1`): Understanding Unicode (1 point)

+ `chr(0)` returns `'\x00'`

+ The string representation (`__repr__()`) shows the escape sequence (`'\x00'`), while the printed representation shows no visible output.

+ In the printed representation (within a call to `print`), this character produces no visible output; in a string representation (the first and third examples), the escape sequence `\x00` appears in the ouptut.

== Problem (`unicode2`): Unicode Encodings (3 points)

+ By starting from a tiny initial vocabulary (256 byte values) and learning efficient merges based on data, we produce a compact, efficient vocabulary. We can then fit more semantic content into a fixed-size context window.

+ The function treats each byte as a Unicode character, but UTF-8 represents characters as sequences of 1-4 bytes. The string `"¢"` would be decoded incorrectly, because the single character's UTF-8 representation is 2 bytes.

+ `0x80 0x80`, because `0x80` (equivalently `10000000`) is a continuation byte, and cannot be used as the first byte in a unicode sequence.