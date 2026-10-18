# Coding AI Development Roadmap

Project: `~/perceptron/stage2/coding_experiment` Current version: V11.4 Next version: V11.5 Updated: October 18, 2026

## 1. Completed milestones

| Version    | Development milestone                                     | Status    |
| ---------- | --------------------------------------------------------- | --------- |
| V1–V9      | Custom neural network and transformer experiments         | Completed |
| V10–V10.10 | Structured code generation and instruction handling       | Completed |
| V11        | Introduced pretrained CodeT5+ 220M                        | Completed |
| V11.1      | Fine-tuned CodeT5+ on 13 algorithms                       | Completed |
| V11.2      | Introduced execution-based generalization testing         | Completed |
| V11.3      | Expanded training to 41 algorithm categories              | Completed |
| V11.4      | Mixed prompt training and validation-selected checkpoints | Completed |

### Current performance

| Benchmark                      | V11.3 | V11.4 |
| ------------------------------ | ----- | ----- |
| Trained algorithms, no hints   | 1/8   | 0/8   |
| Trained algorithms, with hints | 4/8   | 6/8   |
| Unseen algorithms, no hints    | 0/4   | 0/4   |
| Unseen algorithms, with hints  | 0/4   | 0/4   |

Current bottleneck: The model reproduces familiar algorithms but struggles to understand unfamiliar programming instructions.

Training loss is decreasing, but generalization remains poor.

## 2. Updated development roadmap

| Version | Primary objective           | Planned improvements                                                                            |
| ------- | --------------------------- | ----------------------------------------------------------------------------------------------- |
| V11.5   | Dataset expansion           | Thousands of instruction-code examples, diverse programming tasks, consistent prompt formatting |
| V11.6   | Generalization training     | Better sampling, reduced overfitting, algorithm-family holdouts                                 |
| V11.7   | Code verification           | Automatically execute generated programs against test cases                                     |
| V11.8   | Self-correction             | Feed errors back to the model and generate revised implementations                              |
| V11.9   | Multi-step reasoning        | Break complex programming instructions into smaller tasks                                       |
| V12.0   | Autonomous coding assistant | Read project files, propose edits, execute tests, and repair failures with approval             |

These are planned milestones, not implemented features.

## 3. Immediate priority: V11.5

The next version should focus on fixing the training foundation rather than adding more features.

Development targets:

1. Expand the dataset to at least 5,000 diverse instruction-code examples.
2. Keep entire algorithm families out of training for meaningful evaluation.
3. Compare fine-tuned CodeT5+ against the original pretrained model.
4. Match training prompts to actual interactive user requests.
5. Evaluate generated code through execution, not just syntax or loss.
6. Save the best-performing checkpoint without overwriting V11.4.

Proposed success criteria:

| Metric                      | V11.4 baseline | V11.5 target |
| --------------------------- | -------------- | ------------ |
| Trained tasks without hints | 0%             | 50%+         |
| Trained tasks with hints    | 75%            | 85%+         |
| Unseen tasks without hints  | 0%             | 25%+         |
| Unseen tasks with hints     | 0%             | 40%+         |

These are engineering targets, not predicted results. The current four-task unseen benchmark is also too small to establish reliable percentages, so V11.5 should expand it.

## 4. Long-term goal

The intended progression is:

\#chatgpt-mermaid-\_r_da6\_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(237, 237, 237);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-\_r_da6\_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-\_r_da6\_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-\_r_da6\_ .error-icon{fill:rgb(191, 191, 191);}#chatgpt-mermaid-\_r_da6\_ .error-text{fill:rgb(237, 237, 237);stroke:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_da6\_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-\_r_da6\_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-\_r_da6\_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-\_r_da6\_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-\_r_da6\_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-\_r_da6\_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-\_r_da6\_ .marker{fill:rgb(191, 191, 191);stroke:rgb(191, 191, 191);}#chatgpt-mermaid-\_r_da6\_ .marker.cross{stroke:rgb(191, 191, 191);}#chatgpt-mermaid-\_r_da6\_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-\_r_da6\_ p{margin:0;}#chatgpt-mermaid-\_r_da6\_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_da6\_ .cluster-label text{fill:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_da6\_ .cluster-label span{color:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_da6\_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-\_r_da6\_ .label text,#chatgpt-mermaid-\_r_da6\_ span{fill:rgb(237, 237, 237);color:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_da6\_ .node rect,#chatgpt-mermaid-\_r_da6\_ .node circle,#chatgpt-mermaid-\_r_da6\_ .node ellipse,#chatgpt-mermaid-\_r_da6\_ .node polygon,#chatgpt-mermaid-\_r_da6\_ .node path{fill:rgb(143, 157, 178);stroke:rgb(31, 78, 148);stroke-width:1px;}#chatgpt-mermaid-\_r_da6\_ .rough-node .label text,#chatgpt-mermaid-\_r_da6\_ .node .label text,#chatgpt-mermaid-\_r_da6\_ .image-shape .label,#chatgpt-mermaid-\_r_da6\_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-\_r_da6\_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-\_r_da6\_ .rough-node .label,#chatgpt-mermaid-\_r_da6\_ .node .label,#chatgpt-mermaid-\_r_da6\_ .image-shape .label,#chatgpt-mermaid-\_r_da6\_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-\_r_da6\_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-\_r_da6\_ .root .anchor path{fill:rgb(191, 191, 191)!important;stroke-width:0;stroke:rgb(191, 191, 191);}#chatgpt-mermaid-\_r_da6\_ .arrowheadPath{fill:rgb(191, 191, 191);}#chatgpt-mermaid-\_r_da6\_ .edgePath .path{stroke:rgb(191, 191, 191);stroke-width:1px;}#chatgpt-mermaid-\_r_da6\_ .flowchart-link{stroke:rgb(191, 191, 191);fill:none;}#chatgpt-mermaid-\_r_da6\_ .edgeLabel{background-color:rgb(191, 191, 191);text-align:center;}#chatgpt-mermaid-\_r_da6\_ .edgeLabel p{background-color:rgb(191, 191, 191);}#chatgpt-mermaid-\_r_da6\_ .edgeLabel rect{opacity:0.5;background-color:rgb(191, 191, 191);fill:rgb(191, 191, 191);}#chatgpt-mermaid-\_r_da6\_ .labelBkg{background-color:rgba(191, 191, 191, 0.5);}#chatgpt-mermaid-\_r_da6\_ .cluster rect{fill:rgb(191, 191, 191);stroke:rgb(255, 255, 255);stroke-width:1px;}#chatgpt-mermaid-\_r_da6\_ .cluster text{fill:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_da6\_ .cluster span{color:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_da6\_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(191, 191, 191);border:1px solid rgb(255, 255, 255);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-\_r_da6\_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_da6\_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-\_r_da6\_ .icon-shape,#chatgpt-mermaid-\_r_da6\_ .image-shape{background-color:rgb(191, 191, 191);text-align:center;}#chatgpt-mermaid-\_r_da6\_ .icon-shape p,#chatgpt-mermaid-\_r_da6\_ .image-shape p{background-color:rgb(191, 191, 191);padding:2px;}#chatgpt-mermaid-\_r_da6\_ .icon-shape .label rect,#chatgpt-mermaid-\_r_da6\_ .image-shape .label rect{opacity:0.5;background-color:rgb(191, 191, 191);fill:rgb(191, 191, 191);}#chatgpt-mermaid-\_r_da6\_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-\_r_da6\_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-\_r_da6\_ .node .neo-node{stroke:rgb(31, 78, 148);}#chatgpt-mermaid-\_r_da6\_ [data-look="neo"].node rect,#chatgpt-mermaid-\_r_da6\_ [data-look="neo"].cluster rect,#chatgpt-mermaid-\_r_da6\_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-\_r_da6\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_da6\_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-\_r_da6\_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-\_r_da6\_-gradient);stroke-width:1px;}#chatgpt-mermaid-\_r_da6\_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_da6\_ [data-look="neo"].node .neo-line path{stroke:rgb(31, 78, 148);filter:none;}#chatgpt-mermaid-\_r_da6\_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-\_r_da6\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_da6\_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-\_r_da6\_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-\_r_da6\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_da6\_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-\_r_da6\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_da6\_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;}#chatgpt-mermaid-\_r_da6\_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;}#chatgpt-mermaid-\_r_da6\_ .node tspan[font-weight="normal"],#chatgpt-mermaid-\_r_da6\_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-\_r_da6\_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;stroke:rgb(26, 62, 95);stroke-width:1px;}#chatgpt-mermaid-\_r_da6\_ .node rect,#chatgpt-mermaid-\_r_da6\_ .node circle,#chatgpt-mermaid-\_r_da6\_ .node ellipse,#chatgpt-mermaid-\_r_da6\_ .node polygon,#chatgpt-mermaid-\_r_da6\_ .node path{stroke:rgba(255, 255, 255, 0.1);stroke-width:1px;}#chatgpt-mermaid-\_r_da6\_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-\_r_da6\_ .node.mermaid-decision .label-container{stroke:rgb(26, 62, 95);stroke-dasharray:2,2;}#chatgpt-mermaid-\_r_da6\_ .edgePaths .flowchart-link{stroke:rgb(191, 191, 191);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-\_r_da6\_ .marker{stroke:rgb(191, 191, 191);}#chatgpt-mermaid-\_r_da6\_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}V11.4: Fine-tuned CodeT5+V11.5: Larger DatasetV11.6: GeneralizationV11.7: Code VerificationV11.8: Self-CorrectionV11.9: Multi-Step TasksV12.0: Autonomous CodingAssistant

The ultimate objective is an AI that can understand programming requests, generate unfamiliar code, inspect its own results, and correct mistakes rather than merely retrieve memorized implementations.

Next development milestone: V11.5, dataset expansion and genuine generalization training. Preserve `codet5_v114/best/` as the current baseline.&#x20;
