import pandas as pd
import random

diet_types = ['vegan', 'keto', 'vegetarian', 'balanced']
cuisines = ['Indian', 'Mexican', 'Italian', 'Chinese']
allergies = ['dairy', 'nuts', 'gluten', 'none']

users = []

for i in range(1, 101):
    users.append({
        'user_id': i,
        'age': random.randint(18, 50),
        'diet_type': random.choice(diet_types),
        'preferred_cuisine': random.choice(cuisines),
        'allergies': random.choice(allergies),
        'past_orders': random.randint(1, 20),
        'feedback_score': round(random.uniform(3.0, 5.0), 1)
    })

df_users = pd.DataFrame(users)
df_users.to_csv("users.csv", index=False)

meal_types = ['breakfast', 'lunch', 'dinner']
ingredients_list = ['rice', 'lentils', 'spinach', 'pasta', 'chicken', 'tofu', 'beans', 'avocado', 'cheese', 'egg', 'soy sauce']
tags = ['vegan', 'keto', 'vegetarian', 'balanced', 'gluten-free', 'dairy-free', 'protein-rich']

recipes = []
for i in range(101, 151):  # 50 recipes
    recipes.append({
        'recipe_id': i,
        'meal_type': random.choice(meal_types),
        'cuisine': random.choice(cuisines),
        'ingredients': ', '.join(random.sample(ingredients_list, 3)),
        'tags': ', '.join(random.sample(tags, 2)),
        'popularity_score': round(random.uniform(3.5, 5.0), 1)
    })

df_recipes = pd.DataFrame(recipes)
df_recipes.to_csv("recipes.csv", index=False)

from datetime import datetime, timedelta

recommendations = []

for i in range(1, 301):  # 300 recommendations
    user = random.randint(1, 100)
    recipe = random.randint(101, 150)
    clicked = random.choices([0, 1], weights=[0.4, 0.6])[0]
    liked = 1 if clicked and random.random() < 0.7 else 0
    timestamp = datetime(2025, 7, 8) + timedelta(minutes=random.randint(0, 300))
    
    recommendations.append({
        'user_id': user,
        'recipe_id': recipe,
        'clicked': clicked,
        'liked': liked,
        'timestamp': timestamp
    })

df_recs = pd.DataFrame(recommendations)
df_recs.to_csv("recommendations.csv", index=False)

