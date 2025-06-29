import pandas as pd
import streamlit as st
import io
import chardet
import csv
from typing import Dict, Any, List, Tuple, Optional


def detect_file_encoding(file_content: bytes) -> Dict[str, Any]:
    """
    Detect file encoding using chardet with fallback options
    """
    try:
        detection = chardet.detect(file_content)
        detected_encoding = detection.get('encoding', 'utf-8')
        confidence = detection.get('confidence', 0.0)

        # List of encodings to try in order
        encodings_to_try = [
            detected_encoding, 'utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1',
            'cp1252', 'utf-16'
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
            'encodings_to_try': ['utf-8', 'latin-1', 'cp1252'],
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


def process_robust_csv(csv_file) -> pd.DataFrame:
    """
    Process CSV file with robust encoding and separator detection
    """
    try:
        # Reset file pointer and read raw content
        csv_file.seek(0)
        raw_content = csv_file.read()

        st.info(
            f"📄 Processando arquivo: {csv_file.name} ({len(raw_content):,} bytes)"
        )

        # Detect encoding
        encoding_info = detect_file_encoding(raw_content)
        detected_enc = encoding_info.get('detected_encoding', 'utf-8')
        confidence = encoding_info.get('confidence', 0.0)

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
                st.success(f"✅ Decodificado com: {encoding}")
                break
            except UnicodeDecodeError:
                st.warning(f"❌ Falha com encoding: {encoding}")
                continue

        if not decoded_content:
            # Last resort fallback
            decoded_content = raw_content.decode('utf-8', errors='replace')
            used_encoding = 'utf-8 (com substituições)'
            st.warning(
                "⚠️ Usando decodificação com substituição de caracteres")

        # Detect separator
        sep_info = detect_csv_separator(decoded_content)
        detected_sep = sep_info.get('detected_separator', ',')
        st.success(f"📊 Separador detectado: '{detected_sep}'")

        # Show separator analysis
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
                st.success(
                    f"🎉 CSV processado com sucesso! {rows} linhas, {cols} colunas"
                )
                break
            else:
                error_msg = metadata.get('error', 'DataFrame vazio')
                st.warning(
                    f"❌ Falhou com separador '{separator}': {error_msg}")

        if df is None or len(df) == 0:
            st.error("❌ Não foi possível processar o CSV")
            with st.expander("🔍 Preview do arquivo para debug"):
                st.code(decoded_content[:1000])
            raise ValueError(
                "Falha ao processar CSV com todos os separadores testados")

        # Show processing details
        if success_metadata:
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
    column_mapping = {
        # ID variations
        '#': 'id',
        'ticket_id': 'id',
        'issue_id': 'id',
        'numero': 'id',
        'number': 'id',

        # Title/Subject variations
        'título': 'subject',
        'title': 'subject',
        'titulo': 'subject',
        'assunto': 'subject',
        'summary': 'subject',

        # Description variations
        'descrição': 'description',
        'descricao': 'description',
        'desc': 'description',
        'details': 'description',
        'content': 'description',

        # Notes variations
        'últimas notas': 'notes',
        'ultimas notas': 'notes',
        'notas': 'notes',
        'notes': 'notes',
        'last_notes': 'notes',

        # Additional fields
        'responsável': 'assigned_to',
        'responsavel': 'assigned_to',
        'assigned': 'assigned_to',
        'assignee': 'assigned_to',
        'cliente': 'client',
        'customer': 'client',
        'company': 'client',
        'status': 'status',
        'estado': 'status',
        'state': 'status'
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
