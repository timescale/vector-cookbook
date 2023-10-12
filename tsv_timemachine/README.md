# Welcome to the Timescale Time Machine

Chat with the git history of any git repo using OpenAI's GPT.

Stack: 
- UI: Streamlit
- Data framework: LlamaIndex
- Vector database: Timescale Vector (PostgreSQL)

See a [live demo](https://pg-timemachine.streamlit.app/). 
- The live demo is pre-loaded with the history from PostgreSQL. Note that demo disables loading data from other git repos. To load data from the git repo of your choice, launch your own fork by following the instructions below.

## Get a free PostgreSQL database to use with this app
Spin up a [free cloud PostgreSQL database](https://console.cloud.timescale.com/signup?utm_campaign=vectorlaunch&utm_source=github&utm_medium=direct) with Timescale Vector to use in this sample app. You'll get 90 days free by signing up with the link above.

## Launching a fork of Timescale Time Machine
To load data and try the demo on any other git repo, simply:
1) Fork the repo 
2) Launch your own streamlit app. Sign up to Streamlit [here](https://streamlit.io)
3) Set the following [streamlit secrets](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management): 
    - `OPENAI_API_KEY` - Your openAI key.
    - `TIMESCALE_SERVICE_URL` - Your Timescale Service URL. Sign up for a free database [here](https://console.cloud.timescale.com/signup?utm_campaign=vectorlaunch&utm_source=github&utm_medium=direct).
    - `ENABLE_LOAD=1` - Enables Loading Data
