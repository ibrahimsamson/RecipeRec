from recipe_rec.hgnn import RecipeRec
from recipe_rec.dataset import URI_Graph

dataset_folder = '../data/' 


graph = URI_Graph(data_dir="../data")
model = RecipeRec(graph)
user_id = 123
recipe_id = 456
score = model.predict(user_id, recipe_id)


# common function

def get_info_from_model(view_func):

    def wrapper(**kwargs):
        pass
    return wrapper


def get_top_recommendations(user_id, K):
    model = RecipeRec(graph)  # Assuming model and graph are initialized
    scores = {recipe_id: model.predict(user_id, recipe_id) for recipe_id in graph.recipes}
    top_recipes = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:K]
    return [recipe_id for recipe_id, _ in top_recipes]


def recommend_similar_recipes(recipe_id, K):
    model = RecipeRec(graph)
    similar_recipes = model.get_similar_recipes(recipe_id, K)
    return similar_recipes

def train_model(data_path):
    new_graph = URI_Graph(data_dir=data_path)
    model = RecipeRec(new_graph)
    model.train(epochs=10)  # Adjust training parameters as needed
    # Save the trained model for future use




# def recommend_recipes(model, user_id, num_recommendations):
#     """
#     Recommends recipes for a user based on their preferences.

#     Args:
#         model: The RecipeRec model.
#         user_id: The ID of the user.
#         num_recommendations: The number of recipes to recommend.

#     Returns:
#         A list of recommended recipe IDs ranked by score.
#     """
#     all_recipe_ids = model.graph.get_all_recipe_ids()
#     scores = []
#     for recipe_id in all_recipe_ids:
#         scores.append(model.predict(user_id, recipe_id))
#     top_indices = np.argpartition(scores, -num_recommendations)[-num_recommendations:]
#     return [all_recipe_ids[i] for i in top_indices]


# def filter_recommendations(recommended_ids, ingredients):
#     """
#     Filters recommended recipes based on whether they contain desired ingredients.

#     Args:
#         recommended_ids: A list of recommended recipe IDs.
#         ingredients: A list of desired ingredients.

#     Returns:
#         A list of recommended recipe IDs that contain all the desired ingredients.
#     """
#     filtered_ids = []
#     for recipe_id in recommended_ids:
#         recipe_ingredients = model.graph.get_recipe_ingredients(recipe_id)
#         if set(ingredients).issubset(recipe_ingredients):
#         filtered_ids.append(recipe_id)
#     return filtered_ids


# def diversify_recommendations(recommended_ids):
#     """
#     Diversifies recommendations by selecting recipes from different categories.

#     Args:
#         recommended_ids: A list of recommended recipe IDs.

#     Returns:
#         A list of recommended recipe IDs with increased category diversity.
#     """
#     recipe_categories = model.graph.get_recipe_categories(recommended_ids)
#     unique_categories = list(set(recipe_categories))
#     diversified_ids = []
#     for category in unique_categories:
#         category_ids = [i for i, c in zip(recommended_ids, recipe_categories) if c == category]
#         diversified_ids.append(np.random.choice(category_ids))
#     return diversified_ids
