import os
from flask import Flask, send_from_directory # Ensure send_from_directory is imported
from strawberry.flask.views import GraphQLView
import strawberry

# Import the resolvers from a separate file (schema.py)
from schema import Query, Mutation

# Create the schema using strawberry
schema = strawberry.Schema(query=Query, mutation=Mutation)

# Define the static folder for the React app
static_folder_path = os.path.join(os.path.dirname(__file__), 'collaborative-assistant-frontend', 'dist')

app = Flask(__name__, static_folder=static_folder_path, static_url_path='')

# Register the GraphQL endpoint
app.add_url_rule(
    "/graphql",
    view_func=GraphQLView.as_view("graphql_view", schema=schema),
)

# Comment out or remove the old index() route
# @app.route("/")
# def index():
#     return "Collaborative Assistant System API is running. Access /graphql for GraphQL interface."

# Serve React App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        # If the path points to an existing static file, serve it
        return send_from_directory(app.static_folder, path)
    else:
        # Otherwise, serve index.html for client-side routing
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == "__main__":
    # Note: In a production environment, use a production WSGI server like Gunicorn or Waitress
    # and typically set debug=False
    app.run(debug=True)
