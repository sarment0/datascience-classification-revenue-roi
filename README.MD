# Readme Seazone Data Challenge 📜

Hello, this is a repository with a Python script designed to extract data from different datasets, performing a statistical cross-analysis among various sources and providing two outputs to support the thesis argument on the best location for constructing a building with 50 apartments in Itapema - Santa Cataria - Brazil.

## How to run? Instructions ☕️

1. First, clone this repository.
2. You must create the directory ./data and get all the CSV files from this link: https://drive.google.com/drive/folders/1ioYOrQobxsGSC-m2V2fJslcALCh2eFnN
3. in the ./results directory you have the PDF report with my analysis.
4. Install the required modules:

```bash
pip install -r requirements.txt
```

5. to run your own this script you must run this command:

```bash
python app.py
```

6. If you want to run the memory profiler and get the memory usage use the -m or --memory argument

```bash
python app.py --memory
```

7. After you run the project you can check the output directory and open the file map.html it has the demonstration of extracted data in a Map.
