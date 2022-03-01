gsutil cp transform.py gs://ihr-live-workshop/pipeline/transform.py

python pipeline.py --query="SELECT * FROM ihr_live_workshop.transactions" \
  --pipeline-name="ihr-my-pipeline" \
  --pipeline-root="gs://ihr-live-workshop/pipeline/" \
  --project-id="ihr-vertex-pipelines" \
  --temp-location="gs://ihr-live-workshop/tmp/" \
  --region="europe-west4" \
  --service-account="ihr-vertex-live-workshop@ihr-vertex-pipelines.iam.gserviceaccount.com" \
  --transform-file="gs://ihr-live-workshop/pipeline/transform.py"