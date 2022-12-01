from uvicorn import run

from api.app import app

if __name__ == '__main__':
    run(app=app, port=5000)