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
with col2:
    st.title("Sistema de Busca por Similaridade")
    st.caption("Redmine Support Tickets")

# User info in top right
st.markdown("""
<div style="position: absolute; top: 20px; right: 20px; display: flex; align-items: center; gap: 10px;">
    <span>Feito com ❤️ por Saulo Costa</span>
</div>
""",
            unsafe_allow_html=True)

st.markdown("---")

# Layout em colunas para Upload e Busca
col1, col2 = st.columns(2)

# Upload Section
with col1:
    #st.header("📤 Upload de Dados")
    st.header("📤 Importar dados CSV - Redmine")

    #st.markdown("### 📄 Dados CSV do Redmine")
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
            # Process CSV file with robust encoding detection (sem mostrar detalhes)
            tickets_df = process_robust_csv(csv_file, show_details=False)
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

# Search Section
with col2:
    st.header("🔍 Busca por Similaridade")

    # Search input
    description_text = st.text_area(
        "Descrição do chamado",
        placeholder=
        "Descreva o novo chamado para encontrar tickets similares...",
        height=120,
        max_chars=500,
        key="search_description",
        label_visibility="collapsed")

    # Character counter
    char_count = len(description_text) if description_text else 0
    st.caption(f"{char_count}/500 caracteres")

    # Search controls
    has_data_source = (st.session_state.tickets_data is not None)

    search_button = st.button("🔍 Buscar Similaridade",
                              type="primary",
                              use_container_width=True,
                              disabled=not description_text
                              or not has_data_source,
                              key="search_button")

st.markdown("---")

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
    st.header("📋 Chamados Similares")
    st.caption(f"{len(st.session_state.search_results)} resultados")

    # Comentado: campo de ordenação
    # col1, col2 = st.columns([1, 1])
    # with col1:
    #     st.header("📋 Chamados Similares")
    #     st.caption(f"{len(st.session_state.search_results)} resultados")
    #
    # with col2:
    #     sort_option = st.selectbox(
    #         "🔄 Ordenado por relevância",
    #         ["Relevância", "Data de criação", "Responsável"],
    #         index=0,
    #         key="sort_selectbox")

    # Display results
    for idx, result in enumerate(st.session_state.search_results):
        similarity_score = result['similarity_score']
        ticket_data = result['ticket_data']

        # Comentado: Debug - mostrar dados do ticket
        # if idx == 0:  # Só para o primeiro resultado para debug
        #     with st.expander("🔍 Debug - Dados do ticket"):
        #         st.write("Colunas disponíveis:")
        #         for key, value in ticket_data.items():
        #             st.write(f"- {key}: {value}")

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
                # Tenta várias possibilidades para o título/assunto
                subject = (ticket_data.get('subject')
                           or ticket_data.get('Título')
                           or ticket_data.get('titulo') or 'Sem assunto')
                st.markdown(f"### {similarity_color} {subject}")

            with col2:
                st.markdown(
                    f"**{similarity_color} {int(similarity_score * 100)}%**")
                st.caption(similarity_label)

            # Card content
            col1, col2 = st.columns(2)

            with col1:
                # Descrição
                description = (ticket_data.get('description')
                               or ticket_data.get('Descrição')
                               or ticket_data.get('descricao') or 'N/A')
                if description and str(description).strip() and str(
                        description) != 'N/A':
                    desc_text = str(description)[:200]
                    if len(str(description)) > 200:
                        desc_text += "..."
                    st.markdown(f"**📝 Descrição:** {desc_text}")
                else:
                    st.markdown("**📝 Descrição:** Não disponível")

                # ID
                ticket_id = (ticket_data.get('id') or ticket_data.get('#')
                             or ticket_data.get('ID') or 'N/A')
                st.markdown(f"**🆔 ID:** #{ticket_id}")

                # Autor
                author = (ticket_data.get('author') or ticket_data.get('Autor')
                          or ticket_data.get('autor') or 'N/A')
                st.markdown(f"**👤 Autor:** {author}")

                # Cliente
                client = (ticket_data.get('client')
                          or ticket_data.get('Cliente')
                          or ticket_data.get('cliente') or 'N/A')
                st.markdown(f"**🏢 Cliente:** {client}")

            with col2:
                # Sistema
                system = (ticket_data.get('system')
                          or ticket_data.get('Sistema')
                          or ticket_data.get('sistema') or 'N/A')
                st.markdown(f"**💻 Sistema:** {system}")

                # Data de criação
                created_date = (ticket_data.get('created_date')
                                or ticket_data.get('Criado em')
                                or ticket_data.get('criado_em')
                                or ticket_data.get('created') or 'N/A')
                if created_date and str(created_date) != 'N/A':
                    try:
                        if hasattr(created_date, 'strftime'):
                            created_date = created_date.strftime(
                                '%d/%m/%Y %H:%M')
                        elif len(str(created_date)) > 20:
                            created_date = str(created_date)[:19]
                    except:
                        pass
                st.markdown(f"**📅 Criado em:** {created_date}")

                # Data de conclusão
                completed_date = (ticket_data.get('completed_date')
                                  or ticket_data.get('Concluído')
                                  or ticket_data.get('concluido')
                                  or ticket_data.get('completed') or 'N/A')
                if completed_date and str(completed_date) != 'N/A':
                    try:
                        if hasattr(completed_date, 'strftime'):
                            completed_date = completed_date.strftime(
                                '%d/%m/%Y %H:%M')
                        elif len(str(completed_date)) > 20:
                            completed_date = str(completed_date)[:19]
                    except:
                        pass
                st.markdown(f"**✅ Concluído em:** {completed_date}")

                # Solução/Últimas notas
                solution = (ticket_data.get('solution')
                            or ticket_data.get('Últimas notas')
                            or ticket_data.get('ultimas_notas')
                            or ticket_data.get('notes') or '')
                if solution and str(solution).strip() and str(
                        solution) != 'N/A':
                    solution_text = str(solution)[:150]
                    if len(str(solution)) > 150:
                        solution_text += "..."
                    st.markdown(f"**📋 Últimas notas:** {solution_text}")

                # Link do Redmine
                if ticket_id and str(ticket_id).isdigit():
                    redmine_url = f"https://redmine.totvs.amplis.com.br/issues/{ticket_id}"
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
    4. 📋 Visualize os resultados

    **Formatos aceitos:**
    - CSV: dados dos chamados
    - Layout CSV:
    1. 📋 #
    2. 📋 Título
    3. 📋 Cliente
    4. 📋 Sistema
    5. 📋 Criado em
    6. 📋 Concluído
    7. 📋 Autor
    8. 📋 Descrição
    9. 📋 Últimas notas

    **Tecnologia:**
    - Análise NLP com TF-IDF
    - Busca por similaridade semântica
    """)
