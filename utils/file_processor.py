import pandas as pd
import streamlit as st
import io
import PyPDF2
import chardet
import csv
from typing import Optional, List, Dict, Any, Tuple


def detect_encoding(file_content: bytes) -> Dict[str, Any]:
    """
    Detect encoding of file content using chardet

    Args:
        file_content: Raw bytes content of the file

    Returns:
        Dictionary with encoding detection results
    """
    try:
        # Use chardet to detect encoding
        detection_result = chardet.detect(file_content)

        # List of encodings to try in order of preference
        encodings_to_try = [
            detection_result.get('encoding', 'utf-8'), 'utf-8', 'utf-8-sig',
            'latin-1', 'iso-8859-1', 'cp1252', 'utf-16'
        ]

        # Remove duplicates while preserving order
        encodings_to_try = list(dict.fromkeys(filter(None, encodings_to_try)))

        return {
            'detected_encoding': detection_result.get('encoding', 'unknown'),
            'confidence': detection_result.get('confidence', 0.0),
            'encodings_to_try': encodings_to_try,
            'detection_result': detection_result
        }

    except Exception as e:
        return {
            'detected_encoding': 'utf-8',
            'confidence': 0.0,
            'encodings_to_try': ['utf-8', 'latin-1', 'cp1252'],
            'error': str(e)
        }


def detect_csv_separator(file_content: str,
                         sample_size: int = 1024) -> Dict[str, Any]:
    """
    Detect CSV separator by analyzing file content

    Args:
        file_content: String content of the file
        sample_size: Number of characters to analyze

    Returns:
        Dictionary with separator detection results
    """
    # Get sample of file content
    sample = file_content[:sample_size]

    # Common separators to test
    separators = [',', ';', '\t', '|']
    separator_counts = {}

    for sep in separators:
        # Count occurrences of separator
        count = sample.count(sep)
        separator_counts[sep] = count

    # Try to use csv.Sniffer for more advanced detection
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=',;\t|')
        detected_separator = dialect.delimiter
    except:
        # Fall back to most common separator
        if separator_counts:
            detected_separator = max(separator_counts.keys(),
                                     key=lambda x: separator_counts[x])
        else:
            detected_separator = ','

    most_common_sep = ','
    if separator_counts:
        counts_list = [(sep, count) for sep, count in separator_counts.items()]
        if counts_list:
            most_common_sep = max(counts_list, key=lambda x: x[1])[0]

    return {
        'detected_separator': detected_separator,
        'separator_counts': separator_counts,
        'most_common': most_common_sep
    }


def try_read_csv_with_params(
        file_content: str, encoding: str,
        separator: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Try to read CSV with specific encoding and separator

    Args:
        file_content: String content of the file
        encoding: Encoding used to decode the file
        separator: CSV separator character

    Returns:
        Tuple of (DataFrame, metadata dict)
    """
    try:
        # Convert string content to StringIO
        string_io = io.StringIO(file_content)

        # Try reading with pandas
        df = pd.read_csv(
            string_io,
            sep=separator,
            encoding=None,  # Already decoded
            low_memory=False,
            na_values=['', 'NULL', 'null', 'N/A', 'n/a', 'NA', 'na'],
            keep_default_na=True)

        metadata = {
            'success': True,
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'encoding_used': encoding,
            'separator_used': separator
        }

        return df, metadata

    except Exception as e:
        metadata = {
            'success': False,
            'error': str(e),
            'encoding_used': encoding,
            'separator_used': separator
        }
        return pd.DataFrame(), metadata


def process_csv_file(csv_file) -> pd.DataFrame:
    """
    Process uploaded CSV file with robust encoding and separator detection

    Args:
        csv_file: Uploaded CSV file from Streamlit

    Returns:
        Processed DataFrame with ticket data

    Raises:
        Exception: If file processing fails
    """

    # Step 1: Read raw file content
    try:
        # Get raw bytes content
        csv_file.seek(0)  # Reset file pointer
        raw_content = csv_file.read()

        # Display file info
        st.info(
            f"📄 Arquivo carregado: {csv_file.name} ({len(raw_content)} bytes)")

    except Exception as e:
        raise Exception(f"Erro ao ler arquivo: {str(e)}")

    # Step 2: Detect encoding
    try:
        encoding_info = detect_encoding(raw_content)

        st.success(
            f"🔍 Codificação detectada: {encoding_info['detected_encoding']} "
            f"(confiança: {encoding_info['confidence']:.1%})")

        # Show encoding details in expander
        with st.expander("🔧 Detalhes da detecção de codificação"):
            st.json(encoding_info)

    except Exception as e:
        st.warning(f"⚠️ Erro na detecção de codificação: {str(e)}")
        encoding_info = {'encodings_to_try': ['utf-8', 'latin-1', 'cp1252']}

    # Step 3: Try different encodings to decode file
    decoded_content = None
    successful_encoding = None

    for encoding in encoding_info['encodings_to_try']:
        try:
            decoded_content = raw_content.decode(encoding)
            successful_encoding = encoding
            st.success(
                f"✅ Arquivo decodificado com sucesso usando: {encoding}")
            break
        except UnicodeDecodeError as e:
            st.warning(f"❌ Falha ao decodificar com {encoding}: {str(e)}")
            continue

    if decoded_content is None:
        raise Exception(
            "Não foi possível decodificar o arquivo com nenhuma codificação testada"
        )

    # Step 4: Detect CSV separator
    try:
        separator_info = detect_csv_separator(decoded_content)
        st.success(
            f"📊 Separador detectado: '{separator_info['detected_separator']}'")

        with st.expander("📈 Análise de separadores"):
            st.write("Contagem de separadores encontrados:")
            for sep, count in separator_info['separator_counts'].items():
                sep_name = {
                    ',': 'vírgula',
                    ';': 'ponto e vírgula',
                    '\t': 'tabulação',
                    '|': 'pipe'
                }.get(sep, sep)
                st.write(f"- {sep_name} ({repr(sep)}): {count} ocorrências")

    except Exception as e:
        st.warning(f"⚠️ Erro na detecção de separador: {str(e)}")
        separator_info = {'detected_separator': ','}

    # Step 5: Try to read CSV with detected parameters
    separators_to_try = [
        separator_info['detected_separator'], ',', ';', '\t', '|'
    ]

    # Remove duplicates while preserving order
    separators_to_try = list(dict.fromkeys(separators_to_try))

    df = None
    successful_params = None

    for separator in separators_to_try:
        try:
            if successful_encoding is None or decoded_content is None:
                continue
            df, metadata = try_read_csv_with_params(decoded_content,
                                                    successful_encoding,
                                                    separator)

            if metadata['success'] and len(df) > 0:
                successful_params = metadata
                st.success(
                    f"🎉 CSV lido com sucesso! {metadata['rows']} linhas, {metadata['columns']} colunas"
                )
                break
            else:
                st.warning(
                    f"❌ Falha ao ler com separador '{separator}': {metadata.get('error', 'DataFrame vazio')}"
                )

        except Exception as e:
            st.warning(f"❌ Erro ao tentar separador '{separator}': {str(e)}")
            continue

    if df is None or len(df) == 0:
        # Show file preview for debugging
        st.error("❌ Não foi possível ler o arquivo CSV")
        st.write("🔍 **Preview do conteúdo do arquivo para depuração:**")
        st.code(decoded_content[:500] +
                "..." if len(decoded_content) > 500 else decoded_content)
        raise Exception(
            "Falha ao processar CSV com todos os separadores testados")

    # Step 6: Show successful parsing details
    if successful_params is not None:
        with st.expander("✅ Detalhes do processamento bem-sucedido"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Parâmetros utilizados:**")
                st.write(f"- Codificação: {successful_encoding}")
                st.write(
                    f"- Separador: '{successful_params['separator_used']}'")
                st.write(f"- Linhas: {successful_params['rows']}")
                st.write(f"- Colunas: {successful_params['columns']}")

            with col2:
                st.write("**Colunas encontradas:**")
                for col in successful_params['column_names']:
                    st.write(f"- {col}")

    # Step 7: Show file preview
    with st.expander("👁️ Preview dos primeiros caracteres do arquivo"):
        st.code(decoded_content[:1000] +
                "..." if len(decoded_content) > 1000 else decoded_content)

    # Continue with existing processing logic
    try:

        # Validate required columns (flexible approach)
        required_columns = ['id', 'subject', 'description']
        available_columns = df.columns.tolist()

        # Map common column variations to standard names
        column_mapping = {
            # ID variations
            'ticket_id': 'id',
            'issue_id': 'id',
            'numero': 'id',
            'number': 'id',

            # Subject variations
            'title': 'subject',
            'titulo': 'subject',
            'assunto': 'subject',
            'summary': 'subject',

            # Description variations
            'desc': 'description',
            'descrição': 'description',
            'descricao': 'description',
            'details': 'description',
            'content': 'description',

            # Assigned to variations
            'assigned': 'assigned_to',
            'responsavel': 'assigned_to',
            'responsible': 'assigned_to',
            'assignee': 'assigned_to',

            # Client variations
            'customer': 'client',
            'cliente': 'client',
            'company': 'client',
            'organization': 'client',

            # Date variations
            'created': 'created_date',
            'created_on': 'created_date',
            'date_created': 'created_date',
            'data_criacao': 'created_date',
            'created_at': 'created_date',

            # Status variations
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

        # Apply column mapping
        df = df.rename(columns=column_mapping)

        # Ensure we have the minimum required columns
        missing_columns = []
        if 'id' not in df.columns:
            # Try to create ID from index if not available
            df['id'] = df.index + 1

        if 'subject' not in df.columns:
            missing_columns.append('subject (título/assunto)')

        if 'description' not in df.columns:
            missing_columns.append('description (descrição)')

        if missing_columns:
            raise Exception(
                f"Colunas obrigatórias não encontradas: {', '.join(missing_columns)}"
            )

        # Clean and standardize data
        df = _clean_dataframe(df)

        # Add metadata
        df['upload_timestamp'] = pd.Timestamp.now()

        return df

    except pd.errors.EmptyDataError:
        raise Exception("Arquivo CSV está vazio")
    except pd.errors.ParserError as e:
        raise Exception(f"Erro ao analisar CSV: {str(e)}")
    except Exception as e:
        raise Exception(f"Erro no processamento do arquivo: {str(e)}")


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize DataFrame data

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned DataFrame
    """
    # Remove completely empty rows
    df = df.dropna(how='all')

    # Fill NaN values in text columns with empty strings
    text_columns = [
        'subject', 'description', 'assigned_to', 'client', 'status',
        'priority', 'category'
    ]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('')

    # Clean text fields
    for col in ['subject', 'description']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Ensure ID is string type for consistency
    if 'id' in df.columns:
        df['id'] = df['id'].astype(str)

    # Parse dates if available
    date_columns = ['created_date', 'updated_date', 'closed_date']
    for col in date_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass  # Keep original format if parsing fails

    return df


def process_pdf_file(pdf_file) -> Dict[str, Any]:
    """
    Process uploaded PDF file and extract text content

    Args:
        pdf_file: Uploaded PDF file from Streamlit

    Returns:
        Dictionary with PDF metadata and extracted text

    Raises:
        Exception: If PDF processing fails
    """
    try:
        # Reset file pointer to beginning
        pdf_file.seek(0)
        
        # Read PDF content more safely
        try:
            # Try with PyPDF2 first
            pdf_reader = PyPDF2.PdfReader(pdf_file)
        except Exception as e:
            # Try alternative PDF libraries if PyPDF2 fails
            try:
                import pdfplumber
                with pdfplumber.open(pdf_file) as pdf:
                    text_content = ""
                    for page_num, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text_content += f"\n--- Página {page_num + 1} ---\n{page_text.strip()}\n"
                    
                    return {
                        'filename': pdf_file.name,
                        'num_pages': len(pdf.pages),
                        'text_content': text_content.strip() if text_content.strip() else "Texto não pôde ser extraído",
                        'upload_timestamp': pd.Timestamp.now(),
                        'file_size': len(pdf_file.getvalue()) if hasattr(pdf_file, 'getvalue') else 'N/A'
                    }
            except ImportError:
                # Fallback to simple text extraction
                st.warning("Biblioteca pdfplumber não disponível, usando extração básica")
                raise e

        # Extract text from all pages
        text_content = ""
        total_pages = len(pdf_reader.pages)
        
        if total_pages == 0:
            raise Exception("PDF não contém páginas legíveis")

        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    # Clean and normalize page text
                    import re
                    cleaned_text = re.sub(r'\s+', ' ', page_text.strip())
                    text_content += f"\n--- Página {page_num + 1} ---\n{cleaned_text}\n"
            except Exception as e:
                st.warning(f"Erro ao extrair texto da página {page_num + 1}: {str(e)}")
                continue

        # Final text processing
        if text_content.strip():
            # Remove excessive whitespace and normalize
            import re
            text_content = re.sub(r'\s+', ' ', text_content.strip())
            # Remove special characters but keep important punctuation
            text_content = re.sub(r'[^\w\sáàâãéèêíìîóòôõúùûç.,;:!?()-]', ' ', text_content)
            text_content = re.sub(r'\s+', ' ', text_content).strip()
        else:
            text_content = "Texto não pôde ser extraído do PDF"

        # Get PDF metadata safely
        metadata = {
            'filename': pdf_file.name,
            'num_pages': total_pages,
            'text_content': text_content,
            'upload_timestamp': pd.Timestamp.now(),
            'file_size': len(pdf_file.getvalue()) if hasattr(pdf_file, 'getvalue') else 'N/A'
        }

        # Add PDF info if available
        try:
            if hasattr(pdf_reader, 'metadata') and pdf_reader.metadata:
                metadata.update({
                    'title': str(pdf_reader.metadata.get('/Title', '')),
                    'author': str(pdf_reader.metadata.get('/Author', '')),
                    'creator': str(pdf_reader.metadata.get('/Creator', '')),
                    'creation_date': str(pdf_reader.metadata.get('/CreationDate', '')),
                })
        except Exception as e:
            st.warning(f"Erro ao extrair metadados do PDF: {str(e)}")
            
        return metadata

    except Exception as e:
        error_msg = f"Erro ao processar PDF '{pdf_file.name}': {str(e)}"
        st.error(error_msg)
        raise Exception(error_msg)


def validate_csv_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate CSV structure and provide feedback

    Args:
        df: DataFrame to validate

    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'warnings': [],
        'suggestions': [],
        'statistics': {}
    }

    # Check data quality
    total_rows = len(df)

    # Check for empty descriptions
    if 'description' in df.columns:
        empty_descriptions = df['description'].isna().sum() + (
            df['description'] == '').sum()
        if empty_descriptions > 0:
            validation_result['warnings'].append(
                f"{empty_descriptions} tickets têm descrições vazias ({empty_descriptions/total_rows*100:.1f}%)"
            )

    # Check for duplicate tickets
    if 'id' in df.columns:
        duplicate_ids = df['id'].duplicated().sum()
        if duplicate_ids > 0:
            validation_result['warnings'].append(
                f"{duplicate_ids} IDs duplicados encontrados")

    # Statistics
    validation_result['statistics'] = {
        'total_tickets': total_rows,
        'columns_found': list(df.columns),
        'date_range': _get_date_range(df),
        'top_assignees': _get_top_values(df, 'assigned_to'),
        'top_clients': _get_top_values(df, 'client')
    }

    return validation_result


def _get_date_range(df: pd.DataFrame) -> Dict[str, str]:
    """Get date range from DataFrame"""
    date_info = {}

    for col in ['created_date', 'updated_date']:
        if col in df.columns:
            try:
                dates = pd.to_datetime(df[col], errors='coerce').dropna()
                if not dates.empty:
                    date_info[f'{col}_min'] = str(dates.min().date())
                    date_info[f'{col}_max'] = str(dates.max().date())
            except:
                pass

    return date_info


def _get_top_values(df: pd.DataFrame,
                    column: str,
                    top_n: int = 10) -> List[str]:
    """Get top N values from a column"""
    if column not in df.columns:
        return []

    try:
        top_values = df[column].value_counts().head(top_n)
        return [f"{value} ({count})" for value, count in top_values.items()]
    except:
        return []
