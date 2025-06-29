
import pandas as pd
import streamlit as st
import io
import chardet
import csv
from typing import Dict, Any, List, Tuple, Optional


def detect_file_encoding(file_content: bytes) -> Dict[str, Any]:
    """
    Detect file encoding using chardet with fallback options
    Enhanced for better handling of mixed language content
    """
    try:
        detection = chardet.detect(file_content)
        detected_encoding = detection.get('encoding', 'utf-8')
        confidence = detection.get('confidence', 0.0)

        # Enhanced list of encodings to try, prioritizing common ones for mixed content
        encodings_to_try = [
            detected_encoding, 
            'utf-8', 
            'utf-8-sig', 
            'cp1252',  # Windows encoding, good for mixed content
            'latin1',  # ISO-8859-1, handles accented characters
            'iso-8859-1', 
            'cp850',   # IBM PC encoding
            'utf-16',
            'utf-16le',
            'utf-16be'
        ]

        # Remove None values and duplicates while preserving order
        encodings_to_try = list(
            dict.fromkeys([enc for enc in encodings_to_try if enc]))

        return {
            'detected_encoding': detected_encoding,
            'confidence': confidence,
            'encodings_to_try': encodings_to_try
        }
    except Exception as e:
        return {
            'detected_encoding': 'utf-8',
            'confidence': 0.0,
            'encodings_to_try': ['utf-8', 'cp1252', 'latin1', 'iso-8859-1'],
            'error': str(e)
        }


def detect_csv_separator(content: str) -> Dict[str, Any]:
    """
    Detect CSV separator using multiple methods
    """
    sample = content[:2048]  # Use larger sample
    separators = ['\t', ';', ',', '|', ':']

    # Count occurrences of each separator
    sep_counts = {sep: sample.count(sep) for sep in separators}

    # Try csv.Sniffer
    detected_sep = ','
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters='\t;,|:')
        detected_sep = dialect.delimiter
    except Exception:
        # Fall back to most frequent separator
        if sep_counts:
            max_count = max(sep_counts.values())
            if max_count > 0:
                detected_sep = next(sep for sep, count in sep_counts.items()
                                    if count == max_count)

    return {'detected_separator': detected_sep, 'separator_counts': sep_counts}


def try_read_csv(
        content: str, encoding: str,
        separator: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Attempt to read CSV with given parameters
    """
    try:
        df = pd.read_csv(
            io.StringIO(content),
            sep=separator,
            encoding=None,  # Content already decoded
            low_memory=False,
            na_values=['', 'NULL', 'null', 'N/A', 'n/a', 'NA', 'na', '-'],
            keep_default_na=True,
            skipinitialspace=True)

        return df, {
            'success': True,
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'encoding': encoding,
            'separator': separator
        }
    except Exception as e:
        return None, {
            'success': False,
            'error': str(e),
            'encoding': encoding,
            'separator': separator
        }


def process_robust_csv(csv_file, show_details: bool = False) -> pd.DataFrame:
    """
    Process CSV file with robust encoding and separator detection
    
    Args:
        csv_file: Uploaded CSV file
        show_details: Whether to show detailed processing messages
    """
    try:
        # Reset file pointer and read raw content
        csv_file.seek(0)
        raw_content = csv_file.read()

        if show_details:
            st.info(
                f"📄 Processando arquivo: {csv_file.name} ({len(raw_content):,} bytes)"
            )

        # Detect encoding
        encoding_info = detect_file_encoding(raw_content)
        detected_enc = encoding_info.get('detected_encoding', 'utf-8')
        confidence = encoding_info.get('confidence', 0.0)

        if show_details:
            st.success(
                f"🔍 Codificação detectada: {detected_enc} (confiança: {confidence:.1%})"
            )

        # Try to decode with detected encodings
        decoded_content = ""
        used_encoding = "utf-8"

        for encoding in encoding_info.get('encodings_to_try', ['utf-8']):
            try:
                decoded_content = raw_content.decode(encoding)
                used_encoding = encoding
                if show_details:
                    st.success(f"✅ Decodificado com: {encoding}")
                break
            except UnicodeDecodeError:
                if show_details:
                    st.warning(f"❌ Falha com encoding: {encoding}")
                continue

        if not decoded_content:
            # Last resort fallback
            decoded_content = raw_content.decode('utf-8', errors='replace')
            used_encoding = 'utf-8 (com substituições)'
            if show_details:
                st.warning(
                    "⚠️ Usando decodificação com substituição de caracteres")

        # Detect separator
        sep_info = detect_csv_separator(decoded_content)
        detected_sep = sep_info.get('detected_separator', ',')
        if show_details:
            st.success(f"📊 Separador detectado: '{detected_sep}'")

        # Show separator analysis
        if show_details:
            with st.expander("📈 Análise de separadores"):
                sep_names = {
                    ',': 'vírgula',
                    ';': 'ponto e vírgula',
                    '\t': 'tabulação',
                    '|': 'pipe',
                    ':': 'dois pontos'
                }
                for sep, count in sep_info.get('separator_counts', {}).items():
                    name = sep_names.get(sep, f"'{sep}'")
                    st.write(f"- {name}: {count} ocorrências")

        # Try different separators
        separators_to_try = [detected_sep, '\t', ';', ',', '|']
        separators_to_try = list(
            dict.fromkeys(separators_to_try))  # Remove duplicates

        df = None
        success_metadata = None

        for separator in separators_to_try:
            df, metadata = try_read_csv(decoded_content, used_encoding,
                                        separator)

            if metadata.get('success',
                            False) and df is not None and len(df) > 0:
                success_metadata = metadata
                rows = metadata.get('rows', 0)
                cols = metadata.get('columns', 0)
                
                # Sempre mostrar a mensagem de sucesso principal
                st.success(
                    f"🎉 CSV processado com sucesso! {rows} linhas, {cols} colunas"
                )
                break
            else:
                error_msg = metadata.get('error', 'DataFrame vazio')
                if show_details:
                    st.warning(
                        f"❌ Falhou com separador '{separator}': {error_msg}")

        if df is None or len(df) == 0:
            st.error("❌ Não foi possível processar o CSV")
            if show_details:
                with st.expander("🔍 Preview do arquivo para debug"):
                    st.code(decoded_content[:1000])
            raise ValueError(
                "Falha ao processar CSV com todos os separadores testados")

        # Show processing details only if requested
        if show_details and success_metadata:
            with st.expander("✅ Detalhes do processamento"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Parâmetros:**")
                    st.write(f"- Encoding: {used_encoding}")
                    st.write(
                        f"- Separador: '{success_metadata.get('separator', 'N/A')}'"
                    )
                    st.write(f"- Linhas: {success_metadata.get('rows', 0):,}")
                    st.write(
                        f"- Colunas: {success_metadata.get('columns', 0)}")

                with col2:
                    st.write("**Colunas encontradas:**")
                    column_names = success_metadata.get('column_names', [])
                    for col in column_names[:10]:  # Show first 10 columns
                        st.write(f"- {col}")
                    if len(column_names) > 10:
                        st.write(
                            f"... e mais {len(column_names) - 10} colunas")

        # Apply column mapping for Redmine tickets
        df = map_redmine_columns(df)

        # Clean and validate data
        df = clean_dataframe(df)

        return df

    except Exception as e:
        st.error(f"❌ Erro no processamento: {str(e)}")
        raise


def map_redmine_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map common column variations to standard Redmine ticket format
    """
    # Enhanced column mapping for various CSV formats
    column_mapping = {
        # ID variations
        '#': 'id',
        'numero': 'id',
        'number': 'id',
        'ticket_id': 'id',

        # Subject/Title variations
        'título': 'subject',
        'titulo': 'subject',
        'Título': 'subject',
        'title': 'subject',
        'assunto': 'subject',
        'summary': 'subject',

        # Description variations
        'descrição': 'description',
        'descricao': 'description',
        'Descrição': 'description',
        'desc': 'description',
        'details': 'description',
        'content': 'description',

        # Author variations
        'autor': 'author',
        'Autor': 'author',
        'criado_por': 'author',
        'created_by': 'author',
        'reporter': 'author',

        # Client variations
        'cliente': 'client',
        'Cliente': 'client',

        # System variations
        'sistema': 'system',
        'Sistema': 'system',

        # Date variations
        'criado_em': 'created_date',
        'criado em': 'created_date',
        'Criado em': 'created_date',
        'created': 'created_date',
        'data_criacao': 'created_date',

        'concluído': 'completed_date',
        'concluido': 'completed_date',
        'Concluído': 'completed_date',
        'completed': 'completed_date',
        'fechado_em': 'completed_date',

        # Solution/Notes variations
        'solução': 'solution',
        'solucao': 'solution',
        'últimas notas': 'solution',
        'Últimas notas': 'solution',
        'ultimas_notas': 'solution',
        'resolution': 'solution',
        'notes': 'solution',

        # Status variations
        'status': 'status',
        'estado': 'status',
        'state': 'status',

        # Priority variations
        'prioridade': 'priority',
        'prior': 'priority',

        # Category variations
        'categoria': 'category',
        'type': 'category',
        'tipo': 'category'
    }

    # Apply case-insensitive mapping
    df_columns = {col.lower(): col for col in df.columns}
    rename_dict = {}

    for old_name, new_name in column_mapping.items():
        old_lower = old_name.lower()
        if old_lower in df_columns:
            rename_dict[df_columns[old_lower]] = new_name

    df = df.rename(columns=rename_dict)

    # Ensure required columns exist
    if 'id' not in df.columns:
        df['id'] = range(1, len(df) + 1)

    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize DataFrame
    """
    # Remove completely empty rows
    df = df.dropna(how='all')

    # Fill NaN values in text columns
    text_cols = [
        'subject', 'description', 'notes', 'assigned_to', 'client', 'status'
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.strip()

    # Ensure ID is string
    if 'id' in df.columns:
        df['id'] = df['id'].astype(str)

    # Add processing timestamp
    df['processed_at'] = pd.Timestamp.now()

    return df
