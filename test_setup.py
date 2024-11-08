from flask_socketio import SocketIO, emit

# Initialize the SocketIO instance
socketio = SocketIO(test_setup)


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    print("Client connected!")
    emit('message', {'data': 'Connected to the server!'})


@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected!")


if __name__ == "__main__":
    socketio.run(test_setup)
