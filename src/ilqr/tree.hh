#pragma once
#include <utils/debug_utils.hh>

#include <algorithm>
#include <list>
#include <memory>

template<typename T>
class Node 
{
public:
    Node(const std::shared_ptr<Node> &parent, const std::shared_ptr<T> &item) : 
        parent_(parent), item_(item) { }

    virtual ~Node() = default;

    void add_child(const std::shared_ptr<Node> &child)
    {
        children_.push_back(child);
    }

    bool remove_child(const std::shared_ptr<Node> &child)
    {
        auto it = children_.begin();
        bool found_child = false;
        while (it != children_.end())
        {
            if (*it == child)
            {
                it = children_.erase(it);
                found_child = true;
                // Continue search in case child was added more than once.
            }
            else
            {
                ++it;
            }
        }
        return found_child;
    }
    
    bool has_child(const std::shared_ptr<Node> &child) const
    {
        for (const auto &Node : children_)
        {
            if (Node == child)
            {
                return true;
            }
        }
        return false;
    }

    bool has_item(const std::shared_ptr<T> &item) const
    {
        if (item_ && item_ == item)
        {
            return true;
        }
        return false;
    }

    size_t num_children() const { return children_.size(); }

    //
    // Accessors
    //
    
    // Shared ptr to the parent node.
    std::shared_ptr<Node> parent() const { return parent_; }
    // Return a copy so others are not holding references to stored shared pointers.
    // (Everyone gets their own shared pointer)
    std::list<std::shared_ptr<Node>> children() const { return children_; }
    // Access the payload in the node.
    std::shared_ptr<T> item() const { return item_; }


private:
    std::shared_ptr<Node> parent_ = nullptr;
    std::list<std::shared_ptr<Node>> children_ = {};

    std::shared_ptr<T> item_ = nullptr;
};

template <typename T>
class Tree
{
public:
    using NodePtr = std::shared_ptr<Node<T>>;
    
    Tree(const std::shared_ptr<T> &root_item)
    {
        root_ = std::make_shared<Node<T>>(nullptr, root_item);
        add_leaf(root_);
    }

    virtual ~Tree() = default;

    NodePtr root() const {return root_;} 

    // Adds a child node to the parent containing the "item" as payload. 
    // The child is then added to the list of leaf nodes.
    // The shared_ptr to the child is returned.
    NodePtr add_child(NodePtr &parent, const std::shared_ptr<T> &item)
    {
        NodePtr child = std::make_shared<Node<T>>(parent, item);
        IS_TRUE(child);
        parent->add_child(child);
        add_leaf(child);
        IS_TRUE(child);
        return child;
    }

    // Return a copy so others are not holding references to stored shared pointers.
    // (Everyone gets their own shared pointer)
    std::list<NodePtr> leaf_nodes() const { return leaves_; }

    // Traverse the tree depth-first and call the ostream operator for each item or payload 
    // in each node.
    std::string display_string(NodePtr node = nullptr, const std::string &spacing="") const
    {
        if (!node)
        {
            node = root_;
        }
        const std::string endl = "\n";

        std::ostringstream oss;
        oss << *node->item() << endl;
        auto children = node->children();

        for (const auto &child : children)
        {
            IS_TRUE(child);
            oss << spacing << "> " << display_string(child, spacing + "  ");
        }

        return oss.str();
    }


private:
    NodePtr root_ = nullptr;
    std::list<NodePtr> leaves_ = {};

    // Adds to the leaves list and cleanups any in the list that have children nodes 
    // (therefore not actually leaves).
    void add_leaf(const NodePtr &node)
    {
        leaves_.push_back(node);
        cleanup_leaves();
    }

    // Removes any items in the leaves list that have children nodes.
    void cleanup_leaves()
    {
        leaves_.erase(
            std::remove_if(leaves_.begin(), leaves_.end(), [](const NodePtr &node)
                {
                    IS_TRUE(node);    
                    return node->num_children() > 0;
                }
            ), 
            leaves_.end()
        );

    }
    
};

