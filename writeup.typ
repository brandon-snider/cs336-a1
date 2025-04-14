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

= 2. BPE Tokenizer

== Problem (`unicode1`): Understanding Unicode (1 point)

+ `chr(0)` returns `'\x00'`

+ The string representation (`__repr__()`) shows the escape sequence (`'\x00'`), while the printed representation shows no visible output.

+ In the printed representation (within a call to `print`), this character produces no visible output; in a string representation (the first and third examples), the escape sequence `\x00` appears in the ouptut.

== Problem (`unicode2`): Unicode Encodings (3 points)

+ By starting from a tiny initial vocabulary (256 byte values) and learning efficient merges based on data, we produce a compact, efficient vocabulary. We can then fit more semantic content into a fixed-size context window.

+ The function treats each byte as a Unicode character, but UTF-8 represents characters as sequences of 1-4 bytes. The string `"¢"` would be decoded incorrectly, because the single character's UTF-8 representation is 2 bytes.

+ `0x80 0x80`, because `0x80` (equivalently `10000000`) is a continuation byte, and cannot be used as the first byte in a unicode sequence.

== Problem (`train_bpe_tinystories`): BPE Training on TinyStories (2 points)

+ Time: 135.32s (0.038h) \
  Memory: 4GB (per Scalene) \
  Longest token: `' accomplishment'`. This makes sense. With a fairly large vocabulary and a dataset of clean English text, one would expect the longest tokens to be long strings of valid English that appear contiguously in the dataset.

+ Pre-tokenization took roughly half of the overall training time (102s). The specific bottleneck is creating a bytes object for each individual character in each regex match to construct the keys in the table of coarse-grained tokens.

== Problem (`train_bpe_expts_owt`): BPE Training on OpenWebText (2 points)

+ Longest token: `b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82'
` which decodes to `ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ`
  
  This makes sense. This repeated pattern is common when documents are double-encoded or improperly decoded, which is common in scraped web content. In fact, this exact byte sequence appear over 4,500 times in the OWT training set.
  
+ The OpenWebText tokenizer achieves a greater compression ratio, but with the tradeoff of having a much larger vocabulary size that enables it to capture domain-specific patterns and web content artifacts. The TinyStories tokenizer specializes in clean, simple English, reflecting the characteristics of its clean, narrow training set in contrast to the diverse, noisy content from the broader internet.

== Problem (`tokenizer_experiments`): Experiments with Tokenizers (4 points)

+ TinyStories tokenizer compression ratio (bytes/token): $4.01$
  OpenWebText tokenizer compression ratio (bytes/token): $4.50$

+ OpenWebText sample, tokenized with TinyStories tokenizer: $3.40$
  The compression ratio is significantly worse than the compression ratio that the same tokenizer achieves on a sample of data from the same distribution on which the tokenizer was trained. Specifically, the OpenWebText/TinyStories compression ratio is $~85%$ of the TinyStories/TinyStories compression ratio.

+ $"Throughput" approx 6.8 times 10^6 "bytes/second" = 6.8 "MB/second"$

  $T_"Pile" approx (825 times 10^9) "/" (6.8 times 10^6) = 121,324 "seconds" approx 33.7 "hours"$

+ `uint16` is appropriate because of our vocabulary sizes. Both vocabulary sizes are $> 2^8$ and $< 2^16$. This means we can't use an 8-bit representation (we'd have token IDs greater than the representation can store) and we don't need more than a 16-bit representation (all token IDs can be expressed in a 16-bit representation). `uint16` is therefore the most memory-efficient choice.

= 3. Transformer Language Model Architecture

== Problem (`transformer_accounting`): LM resource accounting (5 points)

+ Expression for the total parameter count:
  $ p_"total" = p_"embedding" + p_"layers" + p_"ln_final" + p_"lm_head"  $
  Parameter count of the embedding matrix:
  $ p_"embedding" = "vocab_size" times d_"model" = 50,257 times 1,600 = 80,411,200 $
  Parameter count of all transformer blocks:
  $ p_"layers" &= "num_layers" times p_"layer" \
    p_"layer" &= (2 times p_"ln") + p_"attn" + p_"ffn" \ 
    &= (2 times d_"model") + (p_"wqkv" + p_"out_proj") + (p_"w1" + p_"w2" + p_"w3") \
    &= (2 times d_"model") + (d_"model" times 3 times d_"model" + d_"model" times d_"model") + (3 times d_"model" times d_"ff") \
    &= (2 times 1,600) + (1,600 times 3 times 1,600 + 1,600 times 1,600) + (3 times 1,600 times 6,400) \
    &= 40,963,200 \
    p_"layers" &= 48 times 40,963,200 \
    &= 1,966,233,600
  $
  Parameter count of the final layer norm:
  $ p_"ln" = d_"model" = 1600 $
  Parameter count of the LM head:
  $ p_"lm_head" = d_"model" times "vocab_size" = 1,600 times 50,257 = 80,411,200 $
  Final parameter count:
  $ p_"total" &= p_"embedding" + p_"layers" + p_"ln_final" + p_"lm_head" \
    &= 80,411,200 + 1,966,233,600 + 1600 + 80,411,200 \
    &= 2,127,057,600
  $
  Assuming each parameter is represented using single-recision floating point (4 bytes), the memory required to load the model is:
  $ "memory" &= p_"total" times "memory_per_param" \
    &= 2,127,057,600 times 4 \
    &= 8,508,230,400 "bytes" \
    &approx 8.51"GB"
  $ 
  
+ *GPT-2 XL analysis:*

  FLOPs to compute queries, keys, and values across all layers (using a $W_"qkv"$ matrix):
  $ F_"attn_qkv" &= "num_layers" times [2 times d_"model" times (3 times d_"model") times "context_length"] \
    &= 48 times [2 times 1,600 times (3 times 1600) times 1,024] \
    &= 754,974,720,000 $
  
  Computing attention weights ($Q^T K$) across all layers:
  $ F_"attn_weights" &= "num_layers" times (2 times "context_length" times d_"model" times "context_length") \
    &= 48 times (2 times 1,024 times 1,600 times 1,024) \
    &= 161,061,273,600 $

  Computing attention values ($W V$, where $W$ represents normalized attention weights):
  $ F_"attn_values" &= "num_layers" times (2 times "context_length" times "context_length" times d_"model") \
    &= 48 times (2 times 1,024 times 1,024 times 1,600) \
    &= 161,061,273,600 $

  Output projection after the attention operation, across all layers:
  $ F_"attn_out" &= "num_layers" times (2 times d_"model" times d_"model" times "context_length") \
    &= 48 times (2 times 1,600 times 1,600 times 1,024) \
    &= 251,658,240,000 $
  
  Up projection in the FFN, across all layers:
  $ F_"ffn_up" &= "num_layers" times (2 times d_"model" times d_"ff" times "context_length") \
    &= 48 times (2 times 1,600 times 6,400 times 1,024) \
    &= 1,006,632,960,000 $

  Parallel up projection in the FFN (identical):
  $ F_"ffn_parallel" = F_"ffn_up" = 1,006,632,960,000 $

  Down projection in the FFN, across all layers (same $m,n, p$; different order): 
  $ F_"ffn_down" = F_"ffn_up" = 1,006,632,960,000 $

  Final output projection (LM head):
  $ F_"lm_head" &= 2 times "vocab_size" times d_"model" times "context_length" \
    &= 2 times 50,257 times 1,600 times 1,024 \
    &= 164,682,137,600 $

  Total FLOPs:
  $ F_"total" &= F_"attn" + F_"ffn" + F_"lm_head" \ 

    F_"attn" &= F_"attn_qkv" + F_"attn_weights" + F_"attn_values" + F_"attn_out" \
      &= 754,974,720,000 + (2 times 161,061,273,600) + 251,658,240,000 \
      &= 1,328,755,507,200 \
      &approx 1.33 times 10^12 "FLOPs" \

    F_"ffn" &= F_"ffn_up" + F_"ffn_parallel" + F_"ffn_down" \
      &= 3 times 1,006,632,960,000 \
      &= 3,019,898,880,000 \
      &approx 3.02 times 10^12 "FLOPs" \

    F_"total" &= 1,328,755,507,200 + 3,019,898,880,000 + 164,682,137,600 \
      &= 4,513,336,524,800 \
      &approx 4.51 times 10^12 "FLOPs" $
  
  Proportions:
  $ 
    P_"attn_qkv" &= 754,974,720,000 " / " F_"total" approx 16.7% \
    P_"attn_weights" &= 161,061,273,600 " / " F_"total" approx 3.6% \
    P_"attn_values" &= 161,061,273,600 " / " F_"total" approx 3.6% \
    P_"attn_out" &= 251,658,240,000 " / " F_"total" approx 5.6% \
    P_"attn" &= 1,328,755,507,200 " / " F_"total" approx 29% \
    \
    P_"ffn_up" &= 1,006,632,960,000 " / " F_"total" approx 22.3% \
    P_"ffn_parallel" &= 1,006,632,960,000 " / " F_"total" approx 22.3% \
    P_"ffn_down" &= 1,006,632,960,000 " / " F_"total" approx 22.3% \
    P_"ffn" &= 3,019,898,880,000 " / " F_"total" approx 66.9% \
    \
    P_"lm_head" &= 164,682,137,600 " / " F_"total" approx 3.7% \
  $

+ The FFNs require the most FLOPs by far, accounting for roughly $67%$ of the total (with each of the three matrix multiplications in the FFNs contributing equally). The attention blocks are the next most significant, accounting for roughly $29%$ of the total.

+ *GPT-2 small analysis:*
  
  FLOPs to compute queries, keys, and values across all layers (using a $W_"qkv"$ matrix):
  $ F_"attn_qkv" &= "num_layers" times [2 times d_"model" times (3 times d_"model") times "context_length"] \
    &= 12 times [2 times 768 times (3 times 768) times 1,024] \
    &= 43,486,543,872 $

  Computing attention weights ($Q^T K$) across all layers:
  $ F_"attn_weights" &= "num_layers" times (2 times "context_length" times d_"model" times "context_length") \
    &= 12 times (2 times 1,024 times 768 times 1,024) \
    &= 19,327,352,832 $

  Computing attention values ($W V$, where $W$ represents normalized attention weights):
  $ F_"attn_values" &= "num_layers" times (2 times "context_length" times "context_length" times d_"model") \
    &= 12 times (2 times 1,024 times 1,024 times 768) \
    &= 19,327,352,832 $

  Output projection after the attention operation, across all layers:
  $ F_"attn_out" &= "num_layers" times (2 times d_"model" times d_"model" times "context_length") \
    &= 12 times (2 times 768 times 768 times 1,024) \
    &= 14,495,514,624 $
  
  Up projection in the FFN, across all layers:
  $ F_"ffn_up" &= "num_layers" times (2 times d_"model" times d_"ff" times "context_length") \
    &= 12 times (2 times 768 times 3,072 times 1,024) \
    &= 57,982,058,496 $

  Parallel up projection in the FFN (identical):
  $ F_"ffn_parallel" = F_"ffn_up" = 57,982,058,496 $

  Down projection in the FFN, across all layers (same $m,n,p$; different order):
  $ F_"ffn_down" = F_"ffn_up" = 57,982,058,496 $

  Final output projection (LM head):
  $ F_"lm_head" &= 2 times "vocab_size" times d_"model" times "context_length" \
    &= 2 times 50,257 times 768 times 1,024 \
    &= 79,047,426,048 $

  Total FLOPs:
  $ F_"attn" &= F_"attn_qkv" + F_"attn_weights" + F_"attn_values" + F_"attn_out" \
    &= 43,486,543,872 + 19,327,352,832 + 19,327,352,832 + 14,495,514,624 \
    &= 96,636,764,160 \

    F_"ffn" &= F_"ffn_up" + F_"ffn_parallel" + F_"ffn_down" \
    &= 57,982,058,496 + 57,982,058,496 + 57,982,058,496 \
    &= 173,946,175,488 \

    F_"total" &= F_"attn" + F_"ffn" + F_"lm_head" \
    &= 96,636,764,160 + 173,946,175,488 + 79,047,426,048 \
    &= 349,630,365,696 $

  Proportions:

  $ 
  P_"attn_qkv" &= 43,486,543,872 " / " F_"total" approx 12.4% \
  P_"attn_weights" &= 19,327,352,832 " / " F_"total" approx 5.5% \
  P_"attn_values" &= 19,327,352,832 " / " F_"total" approx 5.5% \
  P_"attn_out" &= 14,495,514,624 " / " F_"total" approx 4.1% \
  P_"attn" &= 96,636,764,160 " / " F_"total" approx 27.6% \
  \
  P_"ffn_up" &= 57,982,058,496 " / " F_"total" approx 16.6% \
  P_"ffn_parallel" &= 57,982,058,496 " / " F_"total" approx 16.6% \
  P_"ffn_down" &= 57,982,058,496 " / " F_"total" approx 16.6% \
  P_"ffn" &= 173,946,175,488 " / " F_"total" approx 49.8% \
  \
  P_"lm_head" &= 79,047,426,048 " / " F_"total" approx 22.6% \
$

  *GPT-2 medium analysis:*

  FLOPs to compute queries, keys, and values across all layers (using a $W_"qkv"$ matrix):
  $ F_"attn_qkv" &= "num_layers" times [2 times d_"model" times (3 times d_"model") times "context_length"] \
    &= 24 times [2 times 1,024 times (3 times 1,024) times 1,024] \
    &= 154,618,822,656 $

  Computing attention weights ($Q^T K$) across all layers:
  $ F_"attn_weights" &= "num_layers" times (2 times "context_length" times d_"model" times "context_length") \
    &= 24 times (2 times 1,024 times 1,024 times 1,024) \
    &= 51,539,607,552 $

  Computing attention values ($W V$, where $W$ represents normalized attention weights):
  $ F_"attn_values" &= "num_layers" times (2 times "context_length" times "context_length" times d_"model") \
    &= 24 times (2 times 1,024 times 1,024 times 1,024) \
    &= 51,539,607,552 $

  Output projection after the attention operation, across all layers:
  $ F_"attn_out" &= "num_layers" times (2 times d_"model" times d_"model" times "context_length") \
    &= 24 times (2 times 1,024 times 1,024 times 1,024) \
    &= 51,539,607,552 $
  
  Up projection in the FFN, across all layers:
  $ F_"ffn_up" &= "num_layers" times (2 times d_"model" times d_"ff" times "context_length") \
    &= 24 times (2 times 1,024 times 4,096 times 1,024) \
    &= 206,158,430,208 $

  Parallel up projection in the FFN (identical):
  $ F_"ffn_parallel" = F_"ffn_up" = 206,158,430,208 $

  Down projection in the FFN, across all layers (same $m,n,p$; different order):
  $ F_"ffn_down" = F_"ffn_up" = 206,158,430,208 $

  Final output projection (LM head):
  $ F_"lm_head" &= 2 times "vocab_size" times d_"model" times "context_length" \
    &= 2 times 50,257 times 1,024 times 1,024 \
    &= 105,396,568,064 $

  Total FLOPs:
  $ F_"attn" &= F_"attn_qkv" + F_"attn_weights" + F_"attn_values" + F_"attn_out" \
    &= 154,618,822,656 + 51,539,607,552 + 51,539,607,552 + 51,539,607,552 \
    &= 309,237,645,312 \

    F_"ffn" &= F_"ffn_up" + F_"ffn_parallel" + F_"ffn_down" \
    &= 206,158,430,208 + 206,158,430,208 + 206,158,430,208 \
    &= 618,475,290,624 \

    F_"total" &= F_"attn" + F_"ffn" + F_"lm_head" \
    &= 309,237,645,312 + 618,475,290,624 + 105,396,568,064 \
    &= 1,033,109,504,000 $

  Proportions:

  $ 
    P_"attn_qkv" &= 154,618,822,656 " / " F_"total" approx 15.0% \
    P_"attn_weights" &= 51,539,607,552 " / " F_"total" approx 5.0% \
    P_"attn_values" &= 51,539,607,552 " / " F_"total" approx 5.0% \
    P_"attn_out" &= 51,539,607,552 " / " F_"total" approx 5.0% \
    P_"attn" &= 309,237,645,312 " / " F_"total" approx 29.9% \
    \
    P_"ffn_up" &= 206,158,430,208 " / " F_"total" approx 20.0% \
    P_"ffn_parallel" &= 206,158,430,208 " / " F_"total" approx 20.0% \
    P_"ffn_down" &= 206,158,430,208 " / " F_"total" approx 20.0% \
    P_"ffn" &= 618,475,290,624 " / " F_"total" approx 59.9% \
    \
    P_"lm_head" &= 105,396,568,064 " / " F_"total" approx 10.2% \
  $

  *GPT-2 large analysis:*

  FLOPs to compute queries, keys, and values across all layers (using a $W_"qkv"$ matrix):
  $ F_"attn_qkv" &= "num_layers" times [2 times d_"model" times (3 times d_"model") times "context_length"] \
    &= 36 times [2 times 1,280 times (3 times 1,280) times 1,024] \
    &= 362,387,865,600 $

  Computing attention weights ($Q^T K$) across all layers:
  $ F_"attn_weights" &= "num_layers" times (2 times "context_length" times d_"model" times "context_length") \
    &= 36 times (2 times 1,024 times 1,280 times 1,024) \
    &= 96,636,764,160 $

  Computing attention values ($W V$, where $W$ represents normalized attention weights):
  $ F_"attn_values" &= "num_layers" times (2 times "context_length" times "context_length" times d_"model") \
    &= 36 times (2 times 1,024 times 1,024 times 1,280) \
    &= 96,636,764,160 $

  Output projection after the attention operation, across all layers:
  $ F_"attn_out" &= "num_layers" times (2 times d_"model" times d_"model" times "context_length") \
    &= 36 times (2 times 1,280 times 1,280 times 1,024) \
    &= 120,795,955,200 $
  
  Up projection in the FFN, across all layers:
  $ F_"ffn_up" &= "num_layers" times (2 times d_"model" times d_"ff" times "context_length") \
    &= 36 times (2 times 1,280 times 5,120 times 1,024) \
    &= 483,183,820,800 $

  Parallel up projection in the FFN (identical):
  $ F_"ffn_parallel" = F_"ffn_up" = 483,183,820,800 $

  Down projection in the FFN, across all layers (same $m,n,p$; different order):
  $ F_"ffn_down" = F_"ffn_up" = 483,183,820,800 $

  Final output projection (LM head):
  $ F_"lm_head" &= 2 times "vocab_size" times d_"model" times "context_length" \
    &= 2 times 50,257 times 1,280 times 1,024 \
    &= 131,745,710,080 $

  Total FLOPs:
  $ F_"attn" &= F_"attn_qkv" + F_"attn_weights" + F_"attn_values" + F_"attn_out" \
    &= 362,387,865,600 + 96,636,764,160 + 96,636,764,160 + 120,795,955,200 \
    &= 676,457,349,120 \

    F_"ffn" &= F_"ffn_up" + F_"ffn_parallel" + F_"ffn_down" \
    &= 483,183,820,800 + 483,183,820,800 + 483,183,820,800 \
    &= 1,449,551,462,400 \

    F_"total" &= F_"attn" + F_"ffn" + F_"lm_head" \
    &= 676,457,349,120 + 1,449,551,462,400 + 131,745,710,080 \
    &= 2,257,754,521,600 $

  Proportions:
  $ 
    P_"attn_qkv" &= 362,387,865,600 " / " F_"total" approx 16.1% \
    P_"attn_weights" &= 96,636,764,160 " / " F_"total" approx 4.3% \
    P_"attn_values" &= 96,636,764,160 " / " F_"total" approx 4.3% \
    P_"attn_out" &= 120,795,955,200 " / " F_"total" approx 5.4% \
    P_"attn" &= 676,457,349,120 " / " F_"total" approx 30.0% \
    \
    P_"ffn_up" &= 483,183,820,800 " / " F_"total" approx 21.4% \
    P_"ffn_parallel" &= 483,183,820,800 " / " F_"total" approx 21.4% \
    P_"ffn_down" &= 483,183,820,800 " / " F_"total" approx 21.4% \
    P_"ffn" &= 1,449,551,462,400 " / " F_"total" approx 64.2% \
    \
    P_"lm_head" &= 131,745,710,080 " / " F_"total" approx 5.8% \
  $

  Analysis:

  The FFN computations increasingly dominate as model size increases. The contribution from the LM head is significant (greater than the contribution from attention) at the smallest model size, and diminishes quickly as model size increases.
  
+ The total FLOPs required increases from $4,513,336,524,800$ to $149,522,795,724,800$. The FLOPs for all operations except the attention operation increase linearly in the length of the context window (by a factor of $2^4$, in this case). The FLOPs for the attention operation (both $Q^T K$ and $W V$, where $W$ represents the normalized attention weights) increase quadratically in the length of the context window (by a factor of $2^8$, in this case).

= 4. Training a Transformer LM

== Problem (`learning_rate_tuning`): Tuning the learning rate (1 point)

For learning rates of 1, 1e1, and 1e2, the loss decreases more quickly as the learning rate is increased (reaching ~23.0 with lr=1, and ~10^-23 with lr=100). With a learning rate of 1e3, the loss diverges, reaching ~10^18 by iteration 10, indicating too large a learning rate.

== Problem (`adamwAccounting`): Resource accounting for AdamW (2 points)

+ We express the peak memory requirements in terms of:
  $ V &= "vocab_size" \
    N &= "num_layers" \
    d &= d_"model" \
    d_"ff" &= 4d \
    h &= "num_heads" \
    T &= "context_length" \
    B &= "batch_size" $

  *Parameters:*

  Embeddings: $V times d $

  Each of the $N$ transformer blocks:
  - 2x RMSNorm: $2d$
  - MHA: $W_"qkv" + W_"out" = [d times (3 times d)] + d^2 = 4d^2$: 
  - FFN: $2 times 4d^2 = 8d^2$
  - Total: $12d^2 + 2d$

  Final RMSNorm: $d$
  
  LM head: $d times V$

  Total parameter count: $P = (2 V d) + N(12d^2 + 2d) + d$
  
  Parameter memory:
  $ "ParamMemory" = 4P "bytes" $

  *Optimizer State*

  Each parameter has a first moment and second moment, so:

  $ "AdamMemory" = 2 times (4P) = 8P "bytes" $

  *Gradient Memory*

  We hold one float per parameter, so:

  $ "GradMemory" = 4P "bytes" $

  *Activation Memory*

  Each transformer block:
  - RMSNorm results: $2 times B times T times d $
  - MHA:
    - QKV projections: $3 times B times T times d$
    - Attention scores ($Q^T K$): $B times h times T times T$
    - Softmax over attention scores: $B times h times T times T$
    - Weighted sum of values: $B times T times d$
    - Output projection: $B times T times d$
  - FFN:
    - $W_1$ output: $B times T times 4d = 4 times B times T times d$
    - SiTU activation: $B times T times 4d = 4 times B times T times d$
    - $W_2$ output: $B times T times d$
  - Total: $16 (B T d) + 2(B h T^2)$

  Across all $N$ blocks: $N (16 B T d + 2 B h T^2)$

  Final RMSNorm: $B times T times d$

  Output embedding (TM head): $B times T times V$

  Cross-entropy on logits: $B times T$

  Total activation count: $A = N (16 B T d + 2 B h T^2) + (B T d) + (B T V) + (B T)$

  $ "ActMemory" = 4A "bytes" $

  *Final Peak Memory Expression*

  $ "TotalMemory" &= "ParamMemory" + "AdamMemory" + "GradMemory" + "ActMemory" \
    &= 4P + 8P + 4P + 4A \
    &= 16P + 4A "bytes" \
    &= 16[(2 V d) + N(12d^2 + 2d) + d] + 4[N (16 B T d + 2 B h T^2) + (B T d) + (B T V) + (B T)]
  $

+ $"TotalMemory"(B) = 15,311,904,768B (B) + 26,168,601,600 "bytes" approx 26 "GB"$
  
  We require $"TotalMemory"(B) lt.eq 80 times 10^9 "bytes", B in ZZ$, so:
  $ 15,311,904,768 (B) + 26,168,601,600 &lt.eq 80,000,000,000 \
  => B &lt.eq 3 $

  With 80 GB available, and storing every intermediate value for every layer in float32, our maximum batch size is 3.

+ \@TODO

+ \@TODO

= 7. Experiments

== Problem (`learning_rate`): Tune the learning rate (3 points)

+ \@TODO (hyperparemeter sweep; learning curves; model w/1.45 loss)

+ \@TODO (folk wisdom; edge of stability)

== Problem (`batch_size_experiment`): Batch size variations (1 point)

+ \@TODO (variants I tried; learning curves; comments on findings)

== Problem (`generate`): Generate text (1 point)

+ \@TODO (256-token output from TS model; comments on fluency and factors)

== Problem (`layer_norm_ablation`): Remove RMSNorm and train (1 point)

+ \@TODO (learning curves at prev. optimal LR and new optimal LR; comments on diff.)

== Problem (`pre-norm_ablation`): Implement post-norm and train (1 point)

+ \@TODO (learning curves for post-norm compared to pre-norm; comments on diff.)

== Problem (`no_pos_emb`): Implement NoPE (1 point)

+ \@TODO (learning curves comparing NoPE and RoPE; comments on diff.)

== Problem (`swiglu_ablation`): SwiGLU vs SiLU (1 point)

+ \@TODO (learning curves comparing SwiGLU and SiLU; few sentences on findings)

== Problem (`main_experiment`): Experiment on OWT (2 points)

+ \@TODO (learning curve on OWT; diff. from TS + interpretation; generation+ comments)

== Problem (`leaderboard`): Leaderboard (6 points)

+ \@TODO (final validation loss; associated learning curve; description of what was done)
