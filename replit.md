# Sistema de Busca por Similaridade - Redmine Support Tickets

## Overview

This is a Streamlit-based similarity search system designed for Redmine support tickets. The application allows users to upload CSV or PDF files containing ticket data and perform intelligent similarity searches using TF-IDF vectorization and cosine similarity algorithms. The system is specifically tailored for Portuguese language content with custom stopwords and text processing capabilities.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application framework
- **Layout**: Wide layout with collapsible sidebar
- **Components**: File upload interface, search functionality, and results display
- **State Management**: Streamlit session state for maintaining data persistence across interactions

### Backend Architecture
- **Core Engine**: Custom SimilarityEngine class using scikit-learn
- **Text Processing**: TF-IDF vectorization with Portuguese language optimization
- **File Processing**: Modular file processors for CSV and PDF formats
- **Similarity Algorithm**: Cosine similarity for finding related tickets

### Data Processing Pipeline
1. File upload and validation
2. Data preprocessing and column mapping
3. Text vectorization using TF-IDF
4. Similarity matrix computation
5. Search query processing and ranking

## Key Components

### SimilarityEngine (`utils/similarity_engine.py`)
- **Purpose**: Core similarity computation engine
- **Technology**: scikit-learn TF-IDF with cosine similarity
- **Features**: 
  - Portuguese stopwords filtering
  - N-gram analysis (1-2 grams)
  - Configurable feature limits (5000 max features)
  - Text preprocessing and normalization

### File Processor (`utils/file_processor.py`)
- **Purpose**: Handle different file formats and data validation
- **Supported Formats**: CSV and PDF files
- **Features**:
  - Flexible column mapping for various CSV formats
  - Data validation and error handling
  - Column standardization (id, subject, description mapping)

### Main Application (`app.py`)
- **Purpose**: Streamlit frontend and user interface
- **Features**:
  - File upload interface
  - Session state management
  - User authentication display
  - Search interface and results presentation

## Data Flow

1. **Data Ingestion**: Users upload CSV or PDF files through Streamlit interface
2. **Data Processing**: Files are processed and validated, columns are mapped to standard format
3. **Vectorization**: Text data is converted to TF-IDF vectors using Portuguese-optimized processing
4. **Index Creation**: Similarity matrix is computed and stored in memory
5. **Search Execution**: User queries are vectorized and compared against existing tickets
6. **Results Ranking**: Similar tickets are ranked by cosine similarity scores
7. **Results Display**: Top matches are presented to the user through Streamlit interface

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms (TF-IDF, cosine similarity)
- **PyPDF2**: PDF file processing

### Text Processing
- Custom Portuguese stopwords list
- TF-IDF vectorization with n-gram support
- Cosine similarity computation

## Deployment Strategy

### Local Development
- Streamlit development server
- Session-based state management
- File-based data processing (no persistent storage)

### Production Considerations
- Memory-based similarity engine (suitable for moderate datasets)
- No database persistence (data loaded per session)
- Single-user session isolation through Streamlit's built-in state management

### Scalability Notes
- Current architecture supports moderate dataset sizes
- TF-IDF matrix stored in memory (consider disk-based storage for larger datasets)
- Single-threaded processing (consider parallel processing for performance improvements)

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes

### June 28, 2025 - CSV Processing Enhancements
- **Robust CSV Upload System**: Implemented advanced CSV processing with automatic encoding detection (UTF-8, Latin-1, CP1252, UTF-16, etc.)
- **Smart Separator Detection**: Automatic detection of CSV separators (comma, semicolon, tab, pipe)
- **Enhanced Error Handling**: Detailed diagnostic messages, file preview for debugging, step-by-step processing feedback
- **New CSV Format Support**: Updated to handle new Redmine export format with columns: #, Título, Cliente, Sistema, Criado em, Concluído, Autor, Descrição, Últimas notas
- **Column Mapping**: Intelligent mapping of Portuguese column names to standardized fields
- **Solution Field Integration**: Added support for "Últimas notas" field as solution/resolution information for better similarity matching

### Technical Improvements
- **File Processing**: Created `robust_csv_processor.py` with chardet library for encoding detection
- **UI Enhancements**: Updated results display to show Client, System, Author, and Solution fields
- **Error Resilience**: Comprehensive fallback mechanisms for encoding and separator detection
- **User Feedback**: Real-time processing status with detailed diagnostic information

## Changelog

Changelog:
- June 28, 2025. Initial setup
- June 28, 2025. Enhanced CSV processing with robust encoding detection and new format support