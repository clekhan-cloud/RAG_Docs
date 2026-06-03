import streamlit as st
import pandas as pd


class DashboardUtils:

    # =====================================================
    # SHOW METRICS
    # =====================================================
    def show_metrics(
        self,
        metrics
    ):

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Retrieval",
            metrics.get(
                "retrieval_score",
                0
            )
        )

        col2.metric(
            "Answer",
            metrics.get(
                "answer_score",
                0
            )
        )

        col3.metric(
            "Overlap",
            metrics.get(
                "overlap_score",
                0
            )
        )

        col4.metric(
            "Final",
            metrics.get(
                "final_score",
                0
            )
        )

    # =====================================================
    # SHOW RETRIEVED DOCS
    # =====================================================
    def show_documents(
        self,
        docs,
        retriever
    ):

        st.subheader(
            "📚 Retrieved Context"
        )

        for i, doc in enumerate(docs):

            citation = (
                retriever.format_citation(
                    doc
                )
            )

            with st.expander(
                f"Chunk {i+1} — {citation}"
            ):

                st.write(
                    doc.get(
                        "text",
                        ""
                    )
                )

                st.json(
                    doc.get(
                        "metadata",
                        {}
                    )
                )

                # image viewer
                if (
                    "image_path" in doc
                    and doc["image_path"]
                ):

                    st.image(
                        doc["image_path"],
                        use_container_width=True
                    )

                    st.caption(
                        doc.get(
                            "image_caption",
                            ""
                        )
                    )

    # =====================================================
    # SHOW BENCHMARK RESULTS
    # =====================================================
    def show_benchmark_results(
        self,
        results
    ):

        st.subheader(
            "📊 Benchmark Results"
        )

        df = pd.DataFrame(results)

        st.dataframe(
            df,
            use_container_width=True
        )

    # =====================================================
    # SHOW SOURCE GROUNDING
    # =====================================================
    def show_grounding(
        self,
        docs
    ):

        st.subheader(
            "📍 Source Grounding"
        )

        for i, doc in enumerate(docs):

            meta = doc.get(
                "metadata",
                {}
            )

            st.markdown(f"""
### Chunk {i+1}

📄 Page: {meta.get("page", "NA")}  
🧩 Chunk: {meta.get("chunk_id", "NA")}  
📚 Source: {meta.get("source", "NA")}  
🗂 Content Type: {meta.get("content_type", "text")}

**Preview**

{doc.get("text", "")[:300]}
""")