#include <iostream>
#include <ilqr/tree.hh>
#include <utils/debug_utils.hh>
#include <utils/math_utils.hh>


Eigen::VectorXd simple_poly(const Eigen::VectorXd &x)
{
    IS_EQUAL(x.size(), 2);
    Eigen::VectorXd y(2);
    y(0) = x(0)*x(0);
    y(1) = x(0) + x(1);
    return y;
}

double quadratic_cost(const Eigen::VectorXd &x)
{
    IS_EQUAL(x.size(), 2);
    Eigen::MatrixXd Q(2,2);
    Q << 5, 6, 6, 8;
    double c = x.transpose() * Q * x;
    return c;
}

void test_jacobian()
{
    Eigen::VectorXd x(2);
    x << 1., 2.;
    auto result = math::jacobian(simple_poly, x);
    WARN("Jacobian:\n" << result)
}

void test_hessian()
{
    Eigen::VectorXd x(2);
    x << 1., 2.;
    auto hess = math::hessian(quadratic_cost, x);
    WARN("Hessian:\n" << hess)
}

void test_gradient()
{
    Eigen::VectorXd x(2);
    x << 1., 2.;
    auto grad = math::gradient(quadratic_cost, x);
    WARN("Grad:\n" << grad)
}

void test_tree()
{
    using string = std::string;
    PRINT("Creating root")
    Tree<string> tree(std::make_shared<string>("1"));
    auto root = tree.root();
    PRINT("adding first child of root");
    auto layer1_1 = tree.add_child(root, std::make_shared<string>("1.1"));
    PRINT("adding second child of root");
    auto layer1_2 = tree.add_child(root, std::make_shared<string>("1.2"));

    PRINT("adding first child of 1.1");
    auto layer1_1_1 = tree.add_child(layer1_1, std::make_shared<string>("1.1.1"));
    PRINT("adding second child of 1.1");
    auto layer1_1_2 = tree.add_child(layer1_1, std::make_shared<string>("1.1.2"));

    PRINT("adding first child of 1.2");
    auto layer1_2_1 = tree.add_child(layer1_2, std::make_shared<string>("1.2.1"));

    PRINT("adding third child of root");
    auto layer1_3 = tree.add_child(root, std::make_shared<string>("1.3"));

    PRINT("adding first child of 1.1.1");
    auto layer1_1_1_1 = tree.add_child(layer1_1_1, std::make_shared<string>("1.1.1.1"));

    PRINT("\n===TREE PRINT:===\n" << tree.display_string(root));

}


int main()
{
    test_tree();

    return 0;
}
