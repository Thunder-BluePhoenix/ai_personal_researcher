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
        
        # Ensure report is a string before downloading
        if isinstance(report, dict):
            report_text = str(report)  # Convert dict to string
        else:
            report_text = report
            
        # Now use report_text for the download button
        st.download_button(
            label="Download Report",
            data=report_text,
            file_name="research_report.md",
            mime="text/markdown",
        )

# Updated visualization section for LangGraph 0.3.30
with st.expander("View Agent Interaction Graph"):
    try:
        import graphviz
        
        # Create a manual representation of the agent workflow
        dot = graphviz.Digraph()
        dot.attr(rankdir='LR')  # Left to right layout
        
        # Define nodes - this is the fixed structure of your agent network
        nodes = ["researcher", "summarizer", "fact_checker", "advisor", "reporter", "END"]
        
        # Define edges - this is the flow between agents
        edges = [
            ("researcher", "summarizer"),
            ("summarizer", "fact_checker"),
            ("fact_checker", "researcher", "confidence < 0.7"),
            ("fact_checker", "advisor", "confidence >= 0.7"),
            ("advisor", "reporter"),
            ("reporter", "END")
        ]
        
        # Add nodes to graph
        for node in nodes:
            if node == "researcher":  # Entry point
                dot.node(node, node, style='filled', fillcolor='lightblue')
            elif node == "END":
                dot.node(node, node, shape="doublecircle")
            else:
                dot.node(node, node)
        
        # Add edges to graph
        for edge in edges:
            if len(edge) == 2:  # Simple edge
                dot.edge(edge[0], edge[1])
            else:  # Edge with condition
                dot.edge(edge[0], edge[1], label=edge[2])
        
        # Display the graph
        st.graphviz_chart(dot)
        
    except Exception as e:
        st.error(f"Could not generate graph visualization: {str(e)}")
        st.error("You may need to install graphviz: pip install graphviz")
        
        # Provide a text representation as fallback
        st.markdown("""
        # Agent Interaction Network
        
        ## Agent Flow
        
        1. **Researcher** → Searches for information on the query topic
        2. **Summarizer** → Condenses research into key points
        3. **Fact Checker** → Verifies information and assigns confidence
           - If confidence < 0.7: Return to Researcher
           - If confidence ≥ 0.7: Proceed to Advisor
        4. **Advisor** → Provides recommendations based on verified information
        5. **Reporter** → Compiles final comprehensive report
        """)