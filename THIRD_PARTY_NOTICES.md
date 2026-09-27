# Third-party notices

Third-party code, model artifacts, and market data retain their respective licenses and terms. No license applied to Ripple's original code changes those rights. This repository does not redistribute model weights or bulk raw market archives.

## Kronos

- Upstream: <https://github.com/shiyu-coder/Kronos>
- Inspected upstream revision: `67b630e67f6a18c9e9be918d9b4337c960db1e9a`
- License: MIT, copyright (c) 2025 ShiYu.
- Current live inference uses the Kronos implementation distributed within sktime, with Ripple's local-checkpoint loading adapter.

The separately downloaded model artifacts are pinned to:

| Artifact | Hugging Face repository | Revision |
| --- | --- | --- |
| Kronos-base | [NeoQuasar/Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base) | `2b554741eca47781b64468546e77fef3e85130e6` |
| Tokenizer | [NeoQuasar/Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base) | `0e0117387f39004a9016484a186a908917e22426` |

Both downloaded model cards declare `license: mit`. Artifact ownership and licensing remain with their upstream providers.

### Kronos MIT notice

MIT License

Copyright (c) 2025 ShiYu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Python dependencies

Dependencies are installed separately, not relicensed as Ripple code. The following descriptions were checked against local package metadata; the complete license and bundled-component notices supplied with each distribution take precedence.

| Dependency | License metadata / notice |
| --- | --- |
| sktime | BSD-3-Clause; its bundled Kronos implementation retains the MIT notice above |
| PyTorch | Multiple component licenses: Apache-2.0, Apache-2.0 WITH LLVM-exception, BSD-2-Clause, BSD-3-Clause, BSL-1.0, MIT |
| pandas | BSD-3-Clause |
| NumPy | BSD-3-Clause, with bundled components under 0BSD, MIT, Zlib, and CC0-1.0 |
| einops | MIT |
| huggingface_hub | Apache license metadata |
| safetensors | Apache Software License classifier |
| tqdm | MPL-2.0 AND MIT |

Other optional research dependencies and transitive dependencies retain their upstream notices as installed. This table is not a replacement for those distribution-level notices.

## Market data

Live prices and the historical source data are from Binance public market-data services:

- Public market API: <https://data-api.binance.vision>
- Historical archive: <https://data.binance.vision>
- Archive documentation: <https://github.com/binance/binance-public-data>

The repository includes selected derived statistics and limited replay excerpts for the demonstration. Public API access does not transfer ownership of the underlying data or make it subject to a Ripple source-code license. Use of Binance services and data remains subject to the applicable provider terms. Binance does not endorse this project.
