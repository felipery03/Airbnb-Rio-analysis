### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

All aditional python libraries to run this project are in requirements.txt. The code should run in Python versions 3.*.<p />
To install all dependences, run in comand line:<br />
   <pre>
   <code>
          pip install -m requirements.txt
    </code>
    </pre>


## Project Motivation<a name="motivation"></a>

For this project, I was interestested in using Airbnb data from 2020 to better understand:

1. Which neighbourhoods are most frequented rent using Airbnb's services in rio de janeiro?
2. Which elements influence the price of a rent?
3. What was the economical impact of COVID-19 pandemic in Rio's tourism activities?

The full set of files related to this course are owned by Airbnb. To run Analysis.ipynb is necessary to download all files related Rio de Janeiro in 25th October 2020 and
copy to data/raw/ path.

List of files:
- listings.csv.gz (extract)
- neighbourhoods.geojson
- reviews.csv.gz (extract)

## File Descriptions <a name="files"></a>

The main notebook called `Analysis.ipynb` is in project root, where contains all analysis made.   

There is an additional script in src/ called`utils.py`, it has all functions used in `Analysis.ipynb`.

## Results <a name="results"></a>

The main findings of the code can be found at the post available [here](https://felipery.medium.com/this-is-the-impact-in-rios-tourism-after-months-of-covid-19-quarantine-3c1ba4c14192).

## Licensing, Authors, Acknowledgements <a name="licensing"></a>

All data credit is from Airbnb.  You can find the Licensing for the data and other descriptive information at the Airbnb link available [here](http://insideairbnb.com/get-the-data.html).  Otherwise, use the code as you wish. 
