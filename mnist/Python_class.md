# Python中的类和继承

## 1. 简介
这个文档将介绍如何在Python中使用类（Class）来创建对象，并通过继承来扩展类的功能。我们将构建一个基本的计算器类，该类能执行加、减、乘、除运算，然后创建一个继承自这个基本计算器的高级计算器，增加计算平均值的功能。

## 2. 定义基础类 `Calculator`

`Calculator` 类将支持基本的算术操作：加、减、乘、除。

### 类属性和方法
- **属性**：`num1`, `num2` - 这两个属性用于存储进行运算的数字。
- **方法**：
  - `add()`: 返回两个数的和。
  - `subtract()`: 返回两个数的差。
  - `multiply()`: 返回两个数的积。
  - `divide()`: 返回两个数的商，除数为零时返回错误消息。

### 代码实现

```python
class Calculator:
    def __init__(self, number1, number2):
        self.num1 = number1
        self.num2 = number2

    def add(self):
        return self.num1 + self.num2

    def subtract(self):
        return self.num1 - self.num2

    def multiply(self):
        return self.num1 * self.num2

    def divide(self):
        if self.num2 != 0:
            return self.num1 / self.num2
        else:
            return "Cannot divide by zero"
```

## 3. 定义子类 `AdvancedCalculator`

这个类继承自 `Calculator` 并添加一个计算平均值的方法。

### 类属性和方法
- 继承自 `Calculator` 的所有属性和方法。
- 新增方法 `average()`: 计算并返回两个数的平均值。

### 代码实现

```python
class AdvancedCalculator(Calculator):
    def __init__(self, number1, number2):
        super(AdvancedCalculator, self).__init__(number1, number2)

    def average(self):
        return (self.num1 + self.num2) / 2
```

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
