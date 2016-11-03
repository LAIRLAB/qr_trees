#pragma once
#include <utils/debug_utils.hh>

#include <algorithm>
#include <list>
#include <memory>

namespace data
{


template<typename T>
class Node 
{
public:
    constexpr static int UNSET_DEPTH = -1;

    Node(const std::shared_ptr<Node> &parent, const std::shared_ptr<T> &item, int depth = UNSET_DEPTH) : 
        depth_(depth), parent_(parent), item_(item) { }

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
    // Get the depth of the Node. -1 implies the depth was not set.
    int depth() const { return depth_; }

    //
    // Setters
    // 
    void set_parent(const std::shared_ptr<Node> &parent) { parent_ = parent; }
    void set_item(const std::shared_ptr<T> &item) { item_ = item; }
    void set_depth(int depth) { depth_ = depth; }


private:
    int depth_ = UNSET_DEPTH; // Depth in the tree.

    std::shared_ptr<Node> parent_ = nullptr;
    std::list<std::shared_ptr<Node>> children_ = {};

    std::shared_ptr<T> item_ = nullptr;
};

template <typename T>
class Tree
{
public:
    using NodePtr = std::shared_ptr<Node<T>>;

    Tree() : root_(nullptr) { }

    Tree(const std::shared_ptr<T> &root_item)
    {
        root_ = std::make_shared<Node<T>>(nullptr, root_item, 0);
        add_leaf(root_);
    }

    Tree(const NodePtr &root_node)
    {
        root_ = root_node;
        if (root_)
        {
            // We may need to renumber the depth of the tree if the root node already has children.
            find_leaves(root_, true, 0);
        }
    }

    virtual ~Tree() = default;

    NodePtr root() const {return root_;} 

    // Adds a child node to the parent containing the "item" as payload. 
    // The child is then added to the list of leaf nodes.
    // The shared_ptr to the child is returned.
    NodePtr add_child(NodePtr &parent, const std::shared_ptr<T> &item)
    {
        // Confirm the tree has been initialized.
        IS_TRUE(root_);

        NodePtr child = std::make_shared<Node<T>>(parent, item, parent->depth()+1);
        IS_TRUE(child);
        parent->add_child(child);
        add_leaf(child);
        IS_TRUE(child);
        return child;
    }

    // Erases the node and all it's children recursively.
    void erase(NodePtr &node)
    {
       // Confirm the tree has been initialized.
       IS_TRUE(root_);
       erase_recursive(node); 
       cleanup_leaves();
    }

    // Pops the node, returning a valid subtree.
    Tree pop(NodePtr &node)
    {
        // Confirm the tree has been initialized.
        IS_TRUE(root_);

        // Remove itself from the parent. If the parent has already cleared its 
        // children list, this will do nothing.
        node->parent()->remove_child(node);

        // We may have popped off leaves so clear the leaves list and recurse to 
        // find the leaves again.  
        // (We could alternatively recurse the popped tree and remove those from 
        // the leaves list).
        leaves_.clear();
        find_leaves(root_);

        return Tree(node);
    }

    // Return a copy so others are not holding references to stored shared pointers.
    // (Everyone gets their own shared pointer)
    std::list<NodePtr> leaf_nodes() const 
    { 
        // Confirm the tree has been initialized.
        IS_TRUE(root_);       

        return leaves_; 
    }

    size_t num_leaf_nodes() const { return leaves_.size(); }

    // Call the ostream operator for each item or payload in each node to generate a 
    // printable string representing the tree structure.
    std::string display_string(const NodePtr &node = nullptr) const 
    { 
        // Confirm the tree has been initialized.
        IS_TRUE(root_);       

        if (node)
        {
            return display_string(node, "");
        }
        return display_string(root_, "");
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

    // Erases the node and all it's children recursively.
    void erase_recursive(NodePtr &node)
    {
        auto children = node->children();
        // Clear the shared pointers being held.
        node->children().clear();
        for (auto &child : children)
        {
            erase_recursive(child);
        }
        
        // Remove itself from the parent. If the parent has already cleared its 
        // children list, this will do nothing.
        node->parent()->remove_child(node);

        // Clear the shared_ptr to the parent.
        node->set_parent(nullptr);

        // Clear the shared_ptr to the item payload.
        node->set_item(nullptr);
        remove_from_leaves_list(node);
    }

    // Remove node from leaves list if it is there.
    void remove_from_leaves_list(NodePtr &node)
    {
        leaves_.erase(
            std::remove_if(leaves_.begin(), leaves_.end(), [&node](const NodePtr &node_cmp)
                {
                    IS_TRUE(node_cmp);    
                    return node == node_cmp;
                }
            ), 
            leaves_.end()
        );
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

    // Searches the tree to find leaf nodes and adds them to the leaves list.
    // If renumber is >= 0, assumes start node is of depth renumber and assigns depths to each of the node in
    // the tree following start node.
    void find_leaves(const NodePtr &start, bool renumber = false, int renumber_start = -1) 
    {
        if (renumber)
        {
            start->set_depth(renumber_start);
        }

        if (start->num_children() == 0)
        {
            leaves_.push_back(start);
        }
        else
        {
            const auto children = start->children();
            for (const auto &child : children)
            {
                IS_TRUE(child);
                find_leaves(child, renumber, renumber_start + 1);
            }
        }
    }

    // Traverse the tree recursively depth-first and call the ostream operator for 
    // each item or payload in each node.
    std::string display_string(const NodePtr &node, const std::string &spacing) const
    {
        const std::string endl = "\n";

        std::ostringstream oss;
        oss << "(" << node->depth() << ") " << *node->item() << endl;
        auto children = node->children();

        for (const auto &child : children)
        {
            IS_TRUE(child);
            oss << spacing << "> " <<  display_string(child, spacing + "  ");
        }

        return oss.str();
    }
    
};

} // namespace data.

