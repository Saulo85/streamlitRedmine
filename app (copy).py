import streamlit as st
import pandas as pd
import numpy as np
from utils.similarity_engine import SimilarityEngine
from utils.file_processor import process_pdf_file
from utils.robust_csv_processor import process_robust_csv
import io

# Configure page
st.set_page_config(page_title="Sistema de Busca por Similaridade",
                   page_icon="🔍",
                   layout="wide",
                   initial_sidebar_state="collapsed")

# Initialize session state
if 'tickets_data' not in st.session_state:
    st.session_state.tickets_data = None
if 'similarity_engine' not in st.session_state:
    st.session_state.similarity_engine = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

# Header
col1, col2 = st.columns([1, 8])
with col1:
    st.markdown("🔍")
with col2:
    st.title("Sistema de Busca por Similaridade")
    st.caption("Redmine Support Tickets")

# User info in top right
st.markdown("""
<div style="position: absolute; top: 20px; right: 20px; display: flex; align-items: center; gap: 10px;">
    <span>👤 Analista</span>
</div>
""",
            unsafe_allow_html=True)

st.markdown("---")

# Upload Section
st.header("📤 Upload de Dados")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📄 Dados CSV do Redmine")
    st.caption("Arraste o arquivo ou clique para selecionar")

    csv_file = st.file_uploader(
        "Selecione o arquivo CSV",
        type=['csv'],
        key="csv_upload",
        help="Upload do arquivo CSV contendo os dados dos chamados do Redmine",
        label_visibility="collapsed",
        accept_multiple_files=False)

    if csv_file is not None:
        try:
            # Process CSV file with robust encoding detection
            tickets_df = process_robust_csv(csv_file)
            st.session_state.tickets_data = tickets_df

            # Initialize similarity engine
            st.session_state.similarity_engine = SimilarityEngine(tickets_df)

            st.success(
                f"✅ Arquivo CSV carregado com sucesso! {len(tickets_df)} tickets encontrados."
            )

            # Show preview
            with st.expander("👁️ Visualizar dados carregados"):
                st.dataframe(tickets_df.head(), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Erro ao processar arquivo CSV: {str(e)}")

with col2:
    st.markdown("### 📋 Documentos PDF")
    st.caption("Arquivos complementares")

    pdf_files = st.file_uploader(
        "Selecione arquivos PDF",
        type=['pdf'],
        accept_multiple_files=True,
        key="pdf_upload",
        help="Upload de documentos PDF complementares",
        label_visibility="collapsed")

    if pdf_files:
        st.success(f"✅ {len(pdf_files)} arquivo(s) PDF carregado(s)")
        for pdf_file in pdf_files:
            st.write(f"📄 {pdf_file.name}")

st.markdown("---")

# Search Section
st.header("🔍 Busca por Similaridade")

# Search input
description_text = st.text_area(
    "Descrição do chamado",
    placeholder="Descreva o novo chamado para encontrar tickets similares...",
    height=120,
    max_chars=500,
    key="search_description",
    label_visibility="collapsed")

# Character counter
char_count = len(description_text) if description_text else 0
st.caption(f"{char_count}/500 caracteres")

# Search controls
col1, col2 = st.columns([1, 4])

with col1:
    refine_search = st.button("⚙️ Refinar Busca", use_container_width=True)

with col2:
    search_button = st.button("🔍 Buscar Similaridade",
                              type="primary",
                              use_container_width=True,
                              disabled=not description_text
                              or st.session_state.tickets_data is None)

# Perform search
if search_button and description_text and st.session_state.similarity_engine:
    with st.spinner("Buscando tickets similares..."):
        try:
            results = st.session_state.similarity_engine.find_similar_tickets(
                description_text, top_k=10)
            st.session_state.search_results = results
        except Exception as e:
            st.error(f"❌ Erro na busca: {str(e)}")

# Display results
if st.session_state.search_results is not None:
    st.markdown("---")

    # Results header
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("📋 Chamados Similares")
        st.caption(f"{len(st.session_state.search_results)} resultados")

    with col2:
        sort_option = st.selectbox(
            "🔄 Ordenado por relevância",
            ["Relevância", "Data de criação", "Responsável"],
            index=0)

    # Display results
    for idx, result in enumerate(st.session_state.search_results):
        similarity_score = result['similarity_score']
        ticket_data = result['ticket_data']

        # Determine similarity color
        if similarity_score >= 0.8:
            similarity_color = "🟢"
            similarity_label = "Alta Similaridade"
        elif similarity_score >= 0.6:
            similarity_color = "🔵"
            similarity_label = "Boa Similaridade"
        else:
            similarity_color = "🟡"
            similarity_label = "Similaridade Moderada"

        # Create result card
        with st.container():
            st.markdown(f"""
            <div style="
                border: 1px solid #e0e0e0; 
                border-radius: 8px; 
                padding: 20px; 
                margin: 10px 0;
                background-color: #fafafa;
            ">
            """,
                        unsafe_allow_html=True)

            # Card header
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(
                    f"### {similarity_color} {ticket_data.get('subject', 'Sem assunto')}"
                )
                st.markdown(
                    f"**📌 Assunto:** {ticket_data.get('subject', 'N/A')}")

            with col2:
                st.markdown(
                    f"**{similarity_color} {int(similarity_score * 100)}%**")
                st.caption(similarity_label)

            # Card content
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    f"**📝 Descrição:** {ticket_data.get('description', 'N/A')[:200]}..."
                )
                st.markdown(f"**🆔 ID:** #{ticket_data.get('id', 'N/A')}")
                st.markdown(f"**👤 Autor:** {ticket_data.get('author', 'N/A')}")
                st.markdown(
                    f"**🏢 Cliente:** {ticket_data.get('client', 'N/A')}")

            with col2:
                st.markdown(
                    f"**💻 Sistema:** {ticket_data.get('system', 'N/A')}")
                st.markdown(
                    f"**📅 Criado em:** {ticket_data.get('created_date', 'N/A')}"
                )

                # Show solution if available
                solution = ticket_data.get('solution', '')
                ##if solution and solution.strip():
                if solution and str(solution).strip():
                    st.markdown(f"**✅ Solução:** {str(solution)[:150]}...")
                    ##st.markdown(f"**✅ Solução:** {solution[:150]}...")

                # Redmine link
                if ticket_data.get('id'):
                    redmine_url = f"https://redmine.totvs.amplis.com.br/issues/{ticket_data.get('id')}"
                    st.markdown(f"**🔗 [Ver no Redmine]({redmine_url})**")

            st.markdown("</div>", unsafe_allow_html=True)

# Show instructions if no data loaded
if st.session_state.tickets_data is None:
    st.info(
        "👆 Faça upload de um arquivo CSV com dados do Redmine para começar a busca por similaridade."
    )

# Sidebar information
with st.sidebar:
    st.header("ℹ️ Informações")
    st.markdown("""
    **Como usar:**
    1. 📤 Faça upload do arquivo CSV com dados do Redmine
    2. 📝 Digite a descrição do novo chamado
    3. 🔍 Clique em "Buscar Similaridade"
    4. 📋 Visualize os resultados ordenados por relevância

    **Formatos aceitos:**
    - CSV: dados dos chamados
    - PDF: documentos complementares

    **Tecnologia:**
    - Análise NLP com TF-IDF
    - Busca por similaridade semântica
    """)
