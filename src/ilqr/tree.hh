#pragma once

#include <algorithm>
#include <list>
#include <memory>

template<typename T>
class node 
{
public:
    node(const std::shared_ptr<node> &parent, const std::shared_ptr<T> &item) : 
        parent_(parent, item) { }

    virtual ~node() = default;

    const std::list<std::shared_ptr<node>>& children()
    {
        return children_;
    }

    void add_child(const std::shared_ptr<node> &child)
    {
        children_.push_back(child);
    }

    bool remove_child(const std::shared_ptr<node> &child)
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
        }
        return found_child;
    }
    
    bool has_child(const std::shared_ptr<node> &child)
    {
        for (const auto &node : children_)
        {
            if (node == child)
            {
                return true;
            }
        }
        return false;
    }

    bool has_item(const std::shared_ptr<T> &item)
    {
        if (item_ && item_ == item)
        {
            return true;
        }
        return false;
    }

    size_t num_children() { return children_.size(); }

    std::shared_ptr<T> item() { return item_; }


private:
    std::shared_ptr<node> parent_ = nullptr;
    std::list<std::shared_ptr<node>> children_ = {};

    std::shared_ptr<T> item_ = nullptr;
};

template <typename T>
class tree
{
    using node_ptr = std::shared_ptr<node<T>>;
    
    tree(std::shared_ptr<T> root_item)
    {
        root_ = std::make_shared<node<T>>(nullptr, root_item);
        add_leaf(root_);
    }

    node_ptr root() {return root_;} 

    // Adds a child node to the parent containing the "item" as payload. 
    // The child is then added to the list of leaf nodes.
    // The shared_ptr to the child is returned.
    node_ptr add_child(node_ptr &parent, const std::shared_ptr<T> &item)
    {
        node_ptr child = std::make_shared<node<T>>(parent, item);
        parent->add_child(child);
        add_leaf(child);
        return child;
    }

    const std::list<node_ptr>& leaf_nodes() { return leaves_; }

    virtual ~tree() = default;

private:
    node_ptr root_ = nullptr;
    std::list<node_ptr> leaves_ = {};

    // Adds to the leaves list and cleanups any in the list that have children nodes 
    // (therefore not actually leaves).
    void add_leaf(const node_ptr &node)
    {
        leaves_.push_back(node);
        cleanup_leaves();
    }

    // Removes any items in the leaves list that have children nodes.
    void cleanup_leaves()
    {
        std::remove_if(leaves_.begin(), leaves_.end(), [](const node_ptr &node)
                {
                    return node->num_children() > 0;
                }
            );
    }

    
};

