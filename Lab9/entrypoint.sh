sleep 5
echo "Initializing Airflow database..."
airflow db migrate
airflow db upgrade
echo "Creating Airflow admin user..."
airflow users create \
    --username admin \
    --firstname Airflow \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin || true 
echo "Starting Airflow scheduler..."
airflow scheduler &
echo "Starting Airflow webserver..."
exec airflow webserver --port 8080