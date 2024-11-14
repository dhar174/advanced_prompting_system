from flask import Flask
from strawberry.flask.views import GraphQLView
import strawberry

# Import the resolvers from a separate file (schema.py)
from schema import Query, Mutation

# Create the schema using strawberry
schema = strawberry.Schema(query=Query, mutation=Mutation)

app = Flask(__name__)

# Register the GraphQL endpoint
app.add_url_rule(
    "/graphql",
    view_func=GraphQLView.as_view("graphql_view", schema=schema),
)

# Basic route for API health check
@app.route("/")
def index():
    return "Collaborative Assistant System API is running. Access /graphql for GraphQL interface."

if __name__ == "__main__":
    app.run(debug=True)

