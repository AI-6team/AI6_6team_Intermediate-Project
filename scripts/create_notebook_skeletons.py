import os
import json

def create_notebook(filename, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print(f"Created {filename}")

def get_common_setup_cells():
    return [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 환경 설정 (Setup)\n",
                "프로젝트 루트 경로를 설정하고 필요한 모듈을 임포트합니다."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "import os\n",
                "\n",
                "# 프로젝트 루트 경로 추가 (현재 위치가 notebooks/ 라고 가정)\n",
                "sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))\n",
                "\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "from bidflow.ingest.loader import RFPLoader\n",
                "from bidflow.retrieval.hybrid_search import HybridRetriever\n",
                "from bidflow.eval.ragas_runner import RagasRunner\n",
                "\n",
                "%matplotlib inline"
            ]
        }
    ]

def main():
    os.makedirs("notebooks", exist_ok=True)

    # --- Exp 1: Chunking ---
    cells_exp1 = get_common_setup_cells() + [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Exp-01: Chunking Strategy Optimization\n",
                "## 실험 목표\n",
                "- 문서 분할 크기(Chunk Size)가 검색 재현율(Recall)에 미치는 영향 분석\n",
                "- 가설: 500~1000자가 적절하며, 너무 크면 정밀도가 떨어지고 너무 작으면 문맥이 잘림\n",
                "\n",
                "## 변수\n",
                "- **Independent Variable**: Chunk Size [500, 1000, 2000]\n",
                "- **Dependent Metric**: Context Recall, Context Precision"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 실험 변수 정의\n",
                "CHUNK_SIZES = [500, 1000, 2000]\n",
                "SAMPLE_FILE = \"../data/sample_rfp.pdf\" # 테스트할 파일 경로\n",
                "\n",
                "# 결과 저장용 리스트\n",
                "results = []"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "loader = RFPLoader()\n",
                "runner = RagasRunner()\n",
                "\n",
                "for size in CHUNK_SIZES:\n",
                "    print(f\"\\n[Experiment] Chunk Size: {size}\")\n",
                "    \n",
                "    # 1. 데이터 적재 (Chunking)\n",
                "    # 주의: 실제 파일이 있어야 함. 없으면 에러 발생하므로 파일 경로 확인 필수\n",
                "    if os.path.exists(SAMPLE_FILE):\n",
                "        with open(SAMPLE_FILE, \"rb\") as f:\n",
                "            doc_hash = loader.process_file(f, os.path.basename(SAMPLE_FILE), chunk_size=size)\n",
                "    else:\n",
                "        print(f\"File not found: {SAMPLE_FILE}\")\n",
                "        break\n",
                "    \n",
                "    # 2. 평가 (여기서는 약식으로 Testset 생성 후 평가)\n",
                "    # 실제로는 Golden Dataset을 로드해서 쓰는 것을 권장\n",
                "    # docs = ... (Load docs from DB)\n",
                "    # eval_df = runner.run_eval(testset)\n",
                "    \n",
                "    # Mock Result for Template\n",
                "    score = 0.85 if size == 1000 else 0.80\n",
                "    results.append({\"chunk_size\": size, \"context_recall\": score})\n",
                "    print(f\"-> Score: {score}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. 결과 분석\n",
                "df_results = pd.DataFrame(results)\n",
                "df_results.plot(x=\"chunk_size\", y=\"context_recall\", kind=\"bar\")\n",
                "plt.title(\"Chunk Size vs Context Recall\")\n",
                "plt.show()"
            ]
        }
    ]
    create_notebook("notebooks/exp01_chunking_optimization.ipynb", cells_exp1)

    # --- Exp 2: Retrieval ---
    cells_exp2 = get_common_setup_cells() + [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Exp-02: Retrieval Strategy Optimization\n",
                "## 실험 목표\n",
                "- Hybrid 검색의 가중치(BM25 vs Vector) 최적화\n",
                "- Top-K 개수에 따른 정확도/노이즈 trade-off 분석\n",
                "\n",
                "## 변수\n",
                "- **Independent Variable**: Alpha (BM25 Weight) [0.3, 0.5, 0.7, 0.9]\n",
                "- **Dependent Metric**: MRR, Context Precision"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "ALPHAS = [0.3, 0.5, 0.7, 0.9]\n",
                "TOP_K = 5\n",
                "QUERY = \"입찰 보증금 납부 기한은 언제까지인가?\" # 테스트용 질문"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "results = []\n",
                "\n",
                "for alpha in ALPHAS:\n",
                "    print(f\"\\n[Experiment] Alpha (BM25): {alpha}\")\n",
                "    \n",
                "    weights = [alpha, 1.0 - alpha]\n",
                "    retriever = HybridRetriever(top_k=TOP_K, weights=weights)\n",
                "    \n",
                "    # 검색 실행\n",
                "    retrieved_docs = retriever.invoke(QUERY)\n",
                "    \n",
                "    # 결과 확인 (상위 1개 문서 내용)\n",
                "    top_content = retrieved_docs[0].page_content[:50] if retrieved_docs else \"No result\"\n",
                "    print(f\"-> Top-1: {top_content}...\")\n",
                "    \n",
                "    # 정량 평가 로직 (Golden Dataset과 비교 필요)\n",
                "    # mrr_score = calculate_mrr(retrieved_docs, ground_truth)\n",
                "    mrr_score = 0.9 if alpha == 0.7 else 0.7\n",
                "    results.append({\"alpha\": alpha, \"mrr\": mrr_score})"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 결과 시각화\n",
                "df = pd.DataFrame(results)\n",
                "df.plot(x=\"alpha\", y=\"mrr\", marker=\"o\")\n",
                "plt.title(\"Alpha(BM25) vs MRR\")\n",
                "plt.grid(True)\n",
                "plt.show()"
            ]
        }
    ]
    create_notebook("notebooks/exp02_retrieval_strategy.ipynb", cells_exp2)
    
    # --- Exp 3: Prompt ---
    cells_exp3 = get_common_setup_cells() + [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Exp-03: Prompt Engineering & Extraction\n",
                "## 실험 목표\n",
                "- 프롬프트 전략(Few-shot vs Zero-shot)에 따른 정보 추출 성공률 비교\n",
                "- 필수 항목 누락(Omission Rate) 최소화\n",
                "\n",
                "## 변수\n",
                "- **Independent Variable**: Prompt Strategy [Zero-shot, Few-shot, Chain-of-Thought]\n",
                "- **Dependent Metric**: Slot Omission Rate"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "PROMPT_STRATEGIES = [\"Zero-shot\", \"Few-shot\"]\n",
                "results = []"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from bidflow.extraction.chains import G3Chain\n",
                "\n",
                "# 주의: G3Chain은 현재 코드가 고정되어 있으므로, \n",
                "# 실험을 위해서는 G3Chain 클래스를 상속받거나 수정하여 prompt를 주입해야 합니다.\n",
                "# 본 노트북에서는 개념 증명(PoC) 코드를 작성합니다.\n",
                "\n",
                "for strategy in PROMPT_STRATEGIES:\n",
                "    print(f\"\\n[Experiment] Strategy: {strategy}\")\n",
                "    \n",
                "    # 체인 초기화 (실제로는 strategy에 따른 template 로드)\n",
                "    # chain = G3Chain(prompt_template=strategy_templates[strategy])\n",
                "    chain = G3Chain()\n",
                "    \n",
                "    # 실행\n",
                "    # output = chain.invoke(dummy_text)\n",
                "    \n",
                "    # 결과 기록\n",
                "    omission_rate = 0.05 if strategy == \"Few-shot\" else 0.15\n",
                "    results.append({\"strategy\": strategy, \"omission_rate\": omission_rate})"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df = pd.DataFrame(results)\n",
                "df.plot(x=\"strategy\", y=\"omission_rate\", kind=\"bar\", color=\"orange\")\n",
                "plt.title(\"Prompt Strategy vs Omission Rate (Lower is Better)\")\n",
                "plt.show()"
            ]
        }
    ]
    create_notebook("notebooks/exp03_prompt_engineering.ipynb", cells_exp3)


if __name__ == "__main__":
    main()
