#!/bin/bash

gcloud builds submit --tag gcr.io/cel-streamlit/fip100

gcloud run deploy fip100 \
  --image gcr.io/cel-streamlit/fip100 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --min-instances=0 \
  --port=8501
