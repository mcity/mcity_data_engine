import ast
import _ast


config_file_path = './config/config.py'

UPDATED_WORKFLOWS =    { "embedding_selection": {
        "mode": "compute",
        "parameters": {
            "compute_representativeness": 0.99,
            "compute_unique_images_greedy": 0.01,
            "compute_unique_images_deterministic": 0.99,
            "compute_similar_images": 0.03,
            "neighbour_count": 3,
        },
        "embedding_models": [
            "detection-transformer-torch",
            "zero-shot-detection-transformer-torch",
            "clip-vit-base32-torch",
        ],
    },}

class ConfigVisitor(ast.NodeTransformer):
        def visit_Assign(self, node):
            # Look for the assignment of the variables we want to modify
            if isinstance(node.targets[0], _ast.Name):
                if node.targets[0].id == "SELECTED_WORKFLOW":
                    node.value = ast.Constant(value=["embedding_selection"] )
                elif node.targets[0].id == "WORKFLOWS":
                    node.value = ast.Constant(value=UPDATED_WORKFLOWS)
                elif node.targets[0].id == "WANDB_ACTIVE":
                    node.value = ast.Constant(value=False)
                elif node.targets[0].id == "HF_DO_UPLOAD":
                    node.value = ast.Constant(value=False)                        
                elif node.targets[0].id == "V51_REMOTE":
                    node.value = ast.Constant(value=False)
            return node


if __name__ == "__main__":
    # Transform the AST
  transformer = ConfigVisitor()

  with open(config_file_path, "r") as file:
    content = file.read()

  parsed_ast = ast.parse(content)
  updated_ast = transformer.visit(parsed_ast)

    # Convert AST back to source code

  updated_content = ast.unparse(updated_ast)

  with open(config_file_path, "w") as file:
    file.write(updated_content)

  print("Config file updated successfully.")
