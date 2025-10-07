# AWS Hackathon - Simple App

This is a simple app for the AWS Hackathon. It demonstrates basic functionality and can be run locally or deployed to AWS.

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
```
> Ensure `requirements.txt` lists all needed packages like `boto3` or `python-dotenv`.

## Running the App

```python main.py```

> Replace `main.py` with your app’s entry point if different.

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
