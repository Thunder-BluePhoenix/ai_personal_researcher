# app.py
import streamlit as st
import asyncio
from main import run_research_workflow

st.set_page_config(page_title="AI Research Agent Network", layout="wide")

st.title("🚀 AI Agent Network for Research & Recommendations")

with st.sidebar:
    st.header("About")
    st.write("""
    This application uses a network of AI agents to:
    1. Research a topic
    2. Summarize findings
    3. Fact-check information
    4. Provide recommendations
    5. Generate a comprehensive report
    """)
    
    st.header("Example Queries")
    st.write("- What are the best crypto wallets in 2025 for long-term holding?")
    st.write("- Is creatine effective for improving athletic performance?")
    st.write("- Should I invest in lithium or hydrogen for green energy?")

query = st.text_input("Enter your research question:", "What are the most sustainable solar panels in 2025?")

if st.button("Run Research"):
    with st.spinner("Agents are working on your research..."):
        # Run the workflow
        report = asyncio.run(run_research_workflow(query))
        
        # Display report
        st.markdown("## Research Report")
        st.markdown(report)
        
        # Optional: Add a download button for the report
        st.download_button(
            label="Download Report",
            data=report,
            file_name="research_report.md",
            mime="text/markdown",
        )

# Visualize agent graph
with st.expander("View Agent Interaction Graph"):
    try:
        from graph.workflow import create_agent_workflow
        workflow = create_agent_workflow()
        dot_graph = workflow.get_graph().draw_graphviz()
        st.graphviz_chart(dot_graph)
    except Exception as e:
        st.error(f"Could not generate graph visualization: {str(e)}")