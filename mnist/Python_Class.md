# Python中的类和继承

## 1. 简介
在这个文档中，我们将学习如何在Python中使用类（Class）来创建对象，并通过继承来扩展类的功能。我们将构建一个基本的计算器类，该类能执行加、减、乘、除运算，然后创建一个继承自这个基本计算器的高级计算器，增加计算平均值的功能。

## 2. 定义基础类 `Calculator`

`Calculator` 类将支持基本的算术操作：加、减、乘、除。

### 代码实现

```python
class Calculator:
    def __init__(self, number1, number2):
        self.number1 = number1
        self.number2 = number2

    def add(self):
        return self.number1 + self.number2

    def subtract(self):
        return self.number1 - self.number2

    def multiply(self):
        return self.number1 * self.number2

    def divide(self):
        if self.number2 != 0:
            return self.number1 / self.number2
        else:
            return "Cannot divide by zero"
```

### 功能描述
- `__init__`: 初始化方法，设置两个操作数。
- `add`: 返回两个数的和。
- `subtract`: 返回两个数的差。
- `multiply`: 返回两个数的积。
- `divide`: 返回两个数的商，除数为零时返回错误消息。

## 3. 定义子类 `AdvancedCalculator`

这个类继承自 `Calculator` 并添加一个计算平均值的方法。

### 代码实现

```python
class AdvancedCalculator(Calculator):
    def __init__(self, number1, number2):
        super(AdvancedCalculator, self).__init__(number1, number2)

    def average(self):
        return (self.number1 + self.number2) / 2
```

### 功能描述
- 继承 `Calculator` 的所有方法。
- `average`: 计算并返回两个数的平均值。

## 4. 使用类进行计算

创建 `AdvancedCalculator` 的实例，并使用其方法进行计算。

### 示例代码

```python
# 创建AdvancedCalculator对象
my_calculator = AdvancedCalculator(20, 10)

# 执行基本运算
print("加法结果:", my_calculator.add())
print("减法结果:", my_calculator.subtract())
print("乘法结果:", my_calculator.multiply())
print("除法结果:", my_calculator.divide())

# 执行高级运算
print("平均值结果:", my_calculator.average())
```

## 5. 结论
通过这个文档，我们学习了如何在Python中使用类和继承来构建具有基本和高级功能的计算器。这展示了面向对象编程的强大功能，允许我们轻松扩展已有的代码以添加新功能。
