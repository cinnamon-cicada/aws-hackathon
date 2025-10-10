# AWS Hackathon
Time is of the essence after a natural disaster, especially for survivors. However, EMS responders often face limited resources. This easy-to-use pipeline will help them quickly find likely survivors.

## Workflow
The project is split into 3 stages.

### Stage 1
Destruction analysis using satellite images. A heatmap is generated based on 1) density of buildings and 2) location of destruction.

### Stage 2
Survivor detection using drone images. Points are added to the map if a potential survivor is detected. Via streamlit, the user can see a map of priority areas.

### Stage 3
AI agents synthesize the data from Stages 1 and 2. Here, the user can see auto-generated suggestions for next steps, based on images and analyses at any given moment.

## Data Used
- To generate the heatmaps, real photos from satellites were used. 
- Photos captured by drones, with synthetically generated humans, were used to validate Stage 2. The one in use as of this GitHub commit is `assets/test1.png`

## Installation

Clone the repository:

git clone https://github.com/cinnamon-cicada/aws-hackathon.git
cd aws-hackathon

Switch to the `INSERT_NAME` branch (if needed):

```
git checkout INSERT_NAME
```

Set up Python environment:

```
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```
Install dependencies:

```
pip install -r requirements.txt
sudo yum install -y mesa-libGLU     # On AWS workspace
sudo yum install -y mesa-libGLU     # On AWS workspace
```
## Running the App

1. Run `streamlit run app.py`
2. Insert Mapbox API key 
    1. Get an account here: https://console.mapbox.com/) OR use:
    2. pk.eyJ1IjoiYmx1ZWNpY2FkYSIsImEiOiJjbWdmbzQzM2IwYWI5MnFvcmN3NGlrMW85In0.EVOuYdIm4CNDZdmJr_-GhQ

## Running the App

```python app.py```

## Pulling Updates

Pull latest changes from your branch:

```
git checkout INSERT_NAME
git pull origin INSERT_NAME
```
Pull latest changes from main branch if needed:

```
git checkout main
git pull origin main
```

## Contributing

1. Fork the repo.  
2. Create a branch:

git checkout -b feature-name

3. Make changes and commit:

git add .
git commit -m "Describe your changes"

4. Push to your fork and open a Pull Request.

## License

This project is licensed under the MIT License.


# ASSUMPTIONS
1. We have pictures without cloud
2. We have pictures before and after disaster