#include <ilqr/tree.hh>
#include <utils/debug_utils.hh>

#include <list>
#include <algorithm>

namespace
{
    using TreeType = data::Tree<std::string>;

    // Test function to check if a node is contained in a list. Used to check if a node is in the
    // leaves list and if children were added properly.
    bool contains_node(const std::list<TreeType::NodePtr> &node_list, const TreeType::NodePtr &node)
    {
        return std::any_of(node_list.begin(), node_list.end(), 
                [&node](const TreeType::NodePtr &test_node)
                {
                    return test_node == node;
                });
    }

    // Helper function to print the values inside a list of nodes.
    void print_nodes(const std::list<TreeType::NodePtr> &node_list)
    {
        std::ostringstream oss;
        for (const auto &node : node_list)
        {
            oss << *(node->item()) << ", ";
        }
        PRINT(oss.str());
    }
}

void test_tree()
{
    using string = std::string;
    PRINT("Creating root")
    TreeType tree(std::make_shared<string>("1"));
    auto root = tree.root();
    IS_EQUAL(tree.num_leaf_nodes(), 1);
    IS_TRUE(contains_node(tree.leaf_nodes(), root));

    PRINT("adding first child of root");
    auto layer1_1 = tree.add_child(root, std::make_shared<string>("1.1"));
    IS_EQUAL(tree.num_leaf_nodes(), 1);
    IS_FALSE(contains_node(tree.leaf_nodes(), root));
    IS_TRUE(contains_node(tree.leaf_nodes(), layer1_1));
    IS_TRUE(contains_node(root->children(), layer1_1));

    PRINT("adding second child of root");
    auto layer1_2 = tree.add_child(root, std::make_shared<string>("1.2"));
    IS_EQUAL(tree.num_leaf_nodes(), 2);
    IS_FALSE(contains_node(tree.leaf_nodes(), root));
    IS_TRUE(contains_node(tree.leaf_nodes(), layer1_1));
    IS_TRUE(contains_node(tree.leaf_nodes(), layer1_2));
    IS_TRUE(contains_node(root->children(), layer1_2));

    PRINT("adding first child of 1.1");
    auto layer1_1_1 = tree.add_child(layer1_1, std::make_shared<string>("1.1.1"));
    IS_EQUAL(tree.num_leaf_nodes(), 2);
    IS_FALSE(contains_node(tree.leaf_nodes(), layer1_1));
    IS_TRUE(contains_node(tree.leaf_nodes(), layer1_1_1));
    IS_TRUE(contains_node(tree.leaf_nodes(), layer1_2));
    IS_TRUE(contains_node(layer1_1->children(), layer1_1_1));

    PRINT("adding second child of 1.1");
    auto layer1_1_2 = tree.add_child(layer1_1, std::make_shared<string>("1.1.2"));
    IS_EQUAL(tree.num_leaf_nodes(), 3);
    IS_TRUE(contains_node(tree.leaf_nodes(), layer1_1_1));
    IS_TRUE(contains_node(tree.leaf_nodes(), layer1_1_2));
    IS_TRUE(contains_node(tree.leaf_nodes(), layer1_2));
    IS_TRUE(contains_node(layer1_1->children(), layer1_1_1));
    IS_TRUE(contains_node(layer1_1->children(), layer1_1_2));

    PRINT("adding first child of 1.2");
    auto layer1_2_1 = tree.add_child(layer1_2, std::make_shared<string>("1.2.1"));
    IS_EQUAL(tree.num_leaf_nodes(), 3);
    IS_TRUE(contains_node(tree.leaf_nodes(), layer1_1_1));
    IS_TRUE(contains_node(tree.leaf_nodes(), layer1_1_2));
    IS_TRUE(contains_node(tree.leaf_nodes(), layer1_2_1));

    PRINT("adding third child of root");
    auto layer1_3 = tree.add_child(root, std::make_shared<string>("1.3"));
    IS_EQUAL(tree.num_leaf_nodes(), 4);
    IS_TRUE(contains_node(tree.leaf_nodes(), layer1_1_1));
    IS_TRUE(contains_node(tree.leaf_nodes(), layer1_1_2));
    IS_TRUE(contains_node(tree.leaf_nodes(), layer1_2_1));
    IS_TRUE(contains_node(tree.leaf_nodes(), layer1_3));

    PRINT("adding first child of 1.1.1");
    auto layer1_1_1_1 = tree.add_child(layer1_1_1, std::make_shared<string>("1.1.1.1"));
    IS_EQUAL(tree.num_leaf_nodes(), 4);
    IS_FALSE(contains_node(tree.leaf_nodes(), layer1_1_1));
    IS_TRUE(contains_node(tree.leaf_nodes(), layer1_1_1_1));
    IS_TRUE(contains_node(tree.leaf_nodes(), layer1_1_2));
    IS_TRUE(contains_node(tree.leaf_nodes(), layer1_2_1));
    IS_TRUE(contains_node(tree.leaf_nodes(), layer1_3));

    PRINT("\n===TREE PRINT:===\n" << tree.display_string(root));
    
    PRINT("\n===Partial TREE PRINT:===\n" << tree.display_string(layer1_1));

    // Erase 1.1, 1.1.1, 1.1.1.1, and 1.1.2 from the tree. Leaving 1.2, 1.2.1 and 1.3 left in the
    // tree.
    tree.erase(layer1_1);
    PRINT("\n===After Erased 1.1. TREE PRINT:===\n" << tree.display_string(root));
    IS_EQUAL(tree.num_leaf_nodes(), 2);
    IS_TRUE(contains_node(tree.leaf_nodes(), layer1_2_1));
    IS_TRUE(contains_node(tree.leaf_nodes(), layer1_3));

    // Sub-tree should have 1.2 as root and 1.2.1 child node.
    TreeType subtree = tree.pop(layer1_2);
    IS_EQUAL(subtree.root(), layer1_2);
    IS_EQUAL(subtree.num_leaf_nodes(), 1);
    IS_TRUE(contains_node(subtree.leaf_nodes(), layer1_2_1));
    IS_EQUAL(*subtree.leaf_nodes().front()->item(), "1.2.1");
    IS_EQUAL(tree.num_leaf_nodes(), 1);
    IS_TRUE(contains_node(tree.leaf_nodes(), layer1_3));

    PRINT("\n===After Popped 1.2 TREE PRINT:===\n" << tree.display_string(root));
    PRINT("\n===The Popped 1.2 SUBTREE PRINT:===\n" << subtree.display_string());

    // Should only have the root left.
    tree.erase(layer1_3);
    PRINT("\n===After Erased 1.3 TREE PRINT:===\n" << tree.display_string(root));
    IS_EQUAL(tree.num_leaf_nodes(), 1);
    IS_TRUE(contains_node(tree.leaf_nodes(), root));
}


int main()
{
    test_tree();

    return 0;
}
