# Third-party notices

The source build fetches [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) at commit `23d0d71502d690230131f22b1393ac41d6b4406e`; it does not redistribute the fork. The HIP patch includes limited context from that MIT-licensed source. Its [upstream LICENSE](https://github.com/PrismML-Eng/llama.cpp/blob/master/LICENSE) states:

> MIT License
>
> Copyright (c) 2023-2026 The ggml authors
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

Bonsai 2 GGUF models remain [PrismML's Apache-2.0 release](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf); model data is not committed here. The optional MTP graft method is linked to [sudoingX/bonsai2-small-gpu](https://github.com/sudoingX/bonsai2-small-gpu), not vendored. Its README states Apache 2.0, but its root LICENSE endpoint was absent when checked; confirm any additional redistribution before copying its tools or documentation.
