# Welcome to the Time Machine

This application, allows you to chat with the git history of any git repo using OpenAI.

You can access the live demo [Here](https://pg-timemachine.streamlit.app/). This demo is pre-loaded with the history from PostgreSQL. The demo
disables loading data from other git repos. To load data, launch your own fork.

## Launching a fork.

To load data and try the demo on any other git repo, simply:
1) Fork the repo 
2) Launch your own streamlit app. 
3) Set the following [streamlit secrets](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management): 
    - `OPENAI_API_KEY` - Your openAI key.
    - `TIMESCALE_SERVICE_URL` - Your Timescale Service URL. Sign up for an account [here](https://www.timescale.com/ai).
    - `ENABLE_LOAD=1` - Enables Loading Data


