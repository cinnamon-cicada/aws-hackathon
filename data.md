# Technical Stack Documentation

## Project Overview
This is an AWS Hackathon project for an EMS (Emergency Medical Services) Urgency Map system focused on disaster response in Nashville, TN. The system provides real-time visualization of emergency situations, human detection, and alert management.

## Core Technologies

### Frontend Framework
- **Streamlit** - Primary web application framework for the main dashboard
- **React/TypeScript** - Used for the disaster response demo component (`disaster_response_demo.tsx`)
- **HTML5 Canvas** - For interactive map rendering and visualization
- **CSS3** - Styling and responsive design

### Backend & Data Processing
- **Python 3.12** - Core programming language
- **Asyncio** - Asynchronous programming for real-time monitoring
- **JSON** - Data serialization and configuration

### Computer Vision & AI
- **OpenCV (cv2)** - Computer vision processing and image manipulation
- **AWS Rekognition** - Cloud-based human detection and object recognition
- **NumPy** - Numerical computing and array operations
- **PIL (Pillow)** - Image processing and format conversion
- **Scikit-image** - Advanced image processing algorithms

### Cloud Services & APIs
- **AWS (Amazon Web Services)**
  - **AWS Rekognition** - Human detection service
  - **Boto3** - AWS SDK for Python
- **Mapbox GL JS** - Interactive mapping and geospatial visualization
- **Mapbox API** - Geospatial data and mapping services

### Data Visualization & Mapping
- **Mapbox GL JS v2.15.0** - Interactive web mapping
- **GeoJSON** - Geospatial data format
- **Folium** - Python mapping library
- **Plotly & Plotly Express** - Interactive data visualization
- **Matplotlib** - Static plotting and visualization
- **Seaborn** - Statistical data visualization

### Geographic Data Processing
- **GeoPandas** - Geospatial data manipulation
- **Shapely** - Geometric operations
- **Pandas** - Data manipulation and analysis

### Environment & Configuration
- **Python-dotenv** - Environment variable management
- **Virtual Environment** - Python dependency isolation

### Development Tools
- **Git** - Version control
- **Jupyter Notebook** - Interactive development (pydeck integration)

## Architecture Components

### Main Application (`app.py`)
- Streamlit-based web dashboard
- Real-time map visualization with Mapbox
- Alert system integration
- Human detection heatmap overlay
- Interactive popup information

### Human Detection System (`human_detection.py`)
- AWS Rekognition integration for human detection
- Real-time video frame processing
- Confidence-based detection filtering
- Geographic coordinate mapping
- Heatmap generation

### Alert Management (`alert_system.py`)
- 100-level emergency alert system
- Drone coordinate tracking
- Alert condition monitoring
- Historical alert management
- Asynchronous alert processing

### Drone Simulation (`drone_simulator.py`)
- Simulated drone movement and positioning
- Video frame capture simulation
- Coordinate generation and tracking
- Image processing pipeline

### Data Processing (`density.py`, `utils.py`)
- Population density calculations
- Building data generation
- JSON data export functionality
- Geographic coordinate transformations

### Frontend Demo (`disaster_response_demo.tsx`)
- React-based disaster response interface
- Interactive canvas-based mapping
- Real-time data visualization
- Survivor tracking and status management
- Timeline playback functionality

## Data Formats & Storage
- **GeoJSON** - Geospatial feature data
- **JSON** - Configuration and detection data
- **PNG/JPEG** - Image assets and heatmaps
- **MP4** - Video simulation data
- **Environment files** - AWS credentials and configuration

## Key Features
1. **Real-time Human Detection** - AWS Rekognition-powered detection
2. **Interactive Mapping** - Mapbox-based geospatial visualization
3. **Alert System** - Multi-level emergency alert management
4. **Heatmap Visualization** - Density-based urgency mapping
5. **Drone Simulation** - Simulated aerial data collection
6. **Disaster Response Interface** - React-based emergency management UI

## Dependencies
All dependencies are listed in `requirements.txt` and include:
- Core web framework (Streamlit)
- Computer vision libraries (OpenCV, PIL, scikit-image)
- AWS services (boto3)
- Data processing (pandas, numpy, geopandas)
- Visualization (plotly, matplotlib, seaborn, folium)
- Geographic processing (shapely)
- Environment management (python-dotenv)

## Deployment Considerations
- Requires AWS credentials for Rekognition service
- Mapbox API token needed for mapping functionality
- Python virtual environment recommended
- Streamlit deployment ready
- Cross-platform compatibility (Windows, Linux, macOS)
