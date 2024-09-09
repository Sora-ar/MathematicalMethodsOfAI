import numpy as np
import matplotlib.pyplot as plt


def fitness_function(x, y):
    """
    Обчислює значення функції придатності для двох змінних x та y.
    Функція має вигляд f(x, y) = 2x^2 - y^2, яку потрібно максимізувати.

    :param x: Значення змінної x.
    :param y: Значення змінної y.
    :return: Значення функції f(x, y) для заданих x та y.
    """
    return 2 * x ** 2 - y ** 2


def create_population(pop_size, x_bounds, y_bounds):
    """
    Створює початкову популяцію для генетичного алгоритму.

    :param pop_size: Кількість особин в популяції.
    :param x_bounds: Кортеж з двома значеннями, що представляють межі діапазону для змінної x
    (наприклад, (-1, 5)).
    :param y_bounds: Кортеж з двома значеннями, що представляють межі діапазону для змінної y
    (наприклад, (-3, 1)).
    :return: Двомірний масив (numpy array) розміром (pop_size, 2), де кожен рядок представляє
    окрему особину з випадковими значеннями x і y в заданих межах.
    """
    population = np.zeros((pop_size, 2))
    population[:, 0] = np.random.uniform(x_bounds[0], x_bounds[1], pop_size)
    population[:, 1] = np.random.uniform(y_bounds[0], y_bounds[1], pop_size)
    return population


def select_parents(population, fitness):
    """
    Відбирає батьків для створення нових поколінь за допомогою методу рулетки.
    Функція реалізує відбір на основі ймовірності, яка пропорційна значенню функції придатності.

    :param population: Двомірний масив (numpy array), де кожен рядок представляє окрему особину популяції.
    :param fitness: Одномірний масив (numpy array) значень функції придатності для кожної особини популяції.
    :return: Двомірний масив (numpy array) нових батьків, обраних за допомогою методу рулетки.
    """
    # Мінімальне значення функції придатності, щоб уникнути від'ємних значень
    min_fitness = np.min(fitness)
    if min_fitness < 0:
        fitness = fitness - min_fitness

    total_fitness = np.sum(fitness)
    probabilities = fitness / total_fitness
    parents_indices = np.random.choice(len(population), size=len(population), p=probabilities)
    return population[parents_indices]


def crossover(parents, crossover_rate=0.8):
    """
    Виконує операцію схрещування для створення нового покоління.
    Пара батьківських особин перехрещується з ймовірністю, визначеною параметром crossover_rate.

    :param parents: Двомірний масив (numpy array) розміром (N, 2), де N — кількість батьків,
    а 2 — кількість змінних (x і y).
    :param crossover_rate: Ймовірність, з якою батьківські особини схрещуються для створення нащадків.
    Має значення від 0 до 1.
    :return: Двомірний масив (numpy array) розміром (N, 2), що представляє нове покоління, створене
    шляхом схрещування батьківських особин.
    """
    offspring = np.empty(parents.shape)
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[i + 1]
        if np.random.rand() < crossover_rate:
            cross_point = np.random.randint(1, len(parent1))
            offspring[i] = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
            offspring[i + 1] = np.concatenate([parent2[:cross_point], parent1[cross_point:]])
        else:
            offspring[i], offspring[i + 1] = parent1, parent2
    return offspring


def mutate(offspring, mutation_rate=0.1, x_bounds=(-1, 5), y_bounds=(-3, 1)):
    """
    Виконує мутацію для нащадків з заданою ймовірністю. Мутація змінює значення змінних x і y
    для кожної особини відповідно до вказаних меж.

    :param offspring: Двомірний масив (numpy array) розміром (N, 2), де N — кількість нащадків,
    а 2 — кількість змінних (x і y).
    :param mutation_rate: Ймовірність мутації для кожної змінної особини. Має значення від 0 до 1.
    :param x_bounds: Кортеж з двома значеннями, що представляють межі діапазону для змінної x (наприклад, (-1, 5)).
    :param y_bounds: Кортеж з двома значеннями, що представляють межі діапазону для змінної y (наприклад, (-3, 1)).
    :return: Двомірний масив (numpy array) розміром (N, 2), де N — кількість нащадків, з
    мутацією змінних x та y відповідно до вказаного mutation_rate.
    """
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            offspring[i][0] = np.random.uniform(x_bounds[0], x_bounds[1])
        if np.random.rand() < mutation_rate:
            offspring[i][1] = np.random.uniform(y_bounds[0], y_bounds[1])
    return offspring


# ГA
def genetic_algorithm(pop_size=50, generations=500, x_bounds=(-1, 5), y_bounds=(-3, 1)):
    """
    Реалізує генетичний алгоритм для оптимізації функції. Виконує ітерації генетичного алгоритму для максимізації
    заданої функції, веде статистику про еволюцію популяції і будує графіки результатів.

    :param pop_size: Кількість особин в популяції. Має значення за замовчуванням 50.
    :param generations: Кількість ітерацій (поколінь) для виконання алгоритму. Має значення за замовчуванням 500.
    :param x_bounds: Кортеж з двох значень, що представляють межі діапазону для змінної x (наприклад, (-1, 5)).
    :param y_bounds: Кортеж з двох значень, що представляють межі діапазону для змінної y (наприклад, (-3, 1)).
    :return: Нічого не повертає. Виводить результати на екран і будує графіки.
    """
    population = create_population(pop_size, x_bounds, y_bounds)
    best_values = []
    evolution_data = []
    average_values = []
    x_values = []
    y_values = []

    for generation in range(generations):
        # Оцінка придатності
        fitness = np.array([fitness_function(ind[0], ind[1]) for ind in population])
        best_values.append(np.max(fitness))

        # Значення x та y популяції
        evolution_data.append(population.copy())

        # Відбір батьків
        parents = select_parents(population, fitness)

        # Схрещування для створення нащадків
        offspring = crossover(parents)

        # Мутація
        offspring = mutate(offspring, mutation_rate=0.1, x_bounds=x_bounds, y_bounds=y_bounds)

        # Оновлення популяції
        population = offspring

    # Визначення найкращого рішення
    fitness = np.array([fitness_function(ind[0], ind[1]) for ind in population])
    best_individual = population[np.argmax(fitness)]

    print(f"Найкраща точка: x = {best_individual[0]:.2f}, y = {best_individual[1]:.2f}")
    print(f"Максимальне значення функції: {np.max(fitness):.2f}")
    print("Всі значення функції:")
    for i, value in enumerate(fitness):
        print(f"Значення {i + 1}: {value:.2f}")

    for generation in range(generations):
        fitness = np.array([fitness_function(ind[0], ind[1]) for ind in population])
        average_values.append(np.mean(fitness))

    for generation in range(generations):
        x_values.append(np.mean(population[:, 0]))  # Середнє значення x в популяції
        y_values.append(np.mean(population[:, 1]))  # Середнє значення y в популяції

    # Найкраще значення функції
    plt.plot(best_values)
    plt.xlabel("Ітерації")
    plt.ylabel("Найкраще значення функції")
    plt.title("Генетичний алгоритм: Оптимізація функції")
    plt.show()

    # Середнє значення функції
    plt.plot(average_values)
    plt.xlabel("Ітерації")
    plt.ylabel("Середнє значення функції")
    plt.title("Середнє значення функції на кожній ітерації")
    plt.show()

    # Еволюція популяції в просторі рішень
    generations_to_plot = [0, 100, 250, 499]
    colors = ['r', 'g', 'b', 'purple']
    plt.figure(figsize=(7, 3))
    for idx, generation in enumerate(generations_to_plot):
        population = evolution_data[generation]
        x_vals = population[:, 0]
        y_vals = population[:, 1]
        plt.scatter(x_vals, y_vals, color=colors[idx], label=f"Ітерація {generation}")

    plt.xlabel('Значення x')
    plt.ylabel('Значення y')
    plt.title("Еволюція популяції в просторі рішень")
    plt.legend()
    plt.grid(True)
    plt.show()


genetic_algorithm()
