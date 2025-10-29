import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

class NaSchModel:
    def __init__(self, L, vmax, p_slow, RHDV=0):
        self.L = L
        self.vmax = vmax
        self.p_slow = p_slow
        self.RHDV = RHDV
        
    def initialize(self, density):
        self.N = int(density * self.L)
        if self.N == 0:
            self.N = 1
        # Случайное распределение автомобилей
        self.positions = np.sort(random.sample(range(self.L), self.N))
        # Начинаем с максимальной скорости
        self.velocities = np.ones(self.N, dtype=int) * self.vmax
        self.is_human = np.random.random(self.N) < self.RHDV
        
    def step(self):
        new_velocities = np.copy(self.velocities)
        new_positions = np.zeros(self.N, dtype=int)
        
        # Вычисление расстояний до впереди идущего автомобиля
        gaps = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if i < self.N - 1:
                gaps[i] = self.positions[i + 1] - self.positions[i] - 1
            else:
                gaps[i] = (self.L - self.positions[i] - 1) + self.positions[0]
        
        # Классические правила NaSch
        for i in range(self.N):
            v = self.velocities[i]
            gap = gaps[i]
            
            # 1. Ускорение
            if v < self.vmax:
                v += 1
            
            # 2. Торможение
            if v > gap:
                v = gap
            
            # 3. Случайное торможение (только для человеческих автомобилей)
            if self.is_human[i] and random.random() < self.p_slow and v > 0:
                v -= 1
            
            new_velocities[i] = v
        
        # Движение
        for i in range(self.N):
            new_pos = (self.positions[i] + new_velocities[i]) % self.L
            new_positions[i] = new_pos
        
        # Сортировка для поддержания порядка
        sort_idx = np.argsort(new_positions)
        self.positions = new_positions[sort_idx]
        self.velocities = new_velocities[sort_idx]
        self.is_human = self.is_human[sort_idx]
        
        return np.mean(self.velocities)

def simulate_fundamental_diagram_NaSch():
    """Фундаментальные диаграммы для модели NaSch на одном графике"""
    print("Модель NaSch: построение фундаментальных диаграмм...")
    
    Ncell, Ts, Te, vmax = 1000, 2000, 500, 5
    # Генерируем 250 случайных плотностей от 0.01 до 1.0
    densities = np.random.rand(250) * 0.99 + 0.01
    densities = np.sort(densities)
    
    plt.figure(figsize=(10, 6))
    
    # Случай 1: pslow=0.8, RHDV=0
    flows1 = []
    model1 = NaSchModel(Ncell, vmax, 0.8, 0)
    
    for density in densities:
        model1.initialize(density)
        for _ in range(Ts):
            model1.step()
        
        total_flow = 0
        for _ in range(Te):
            avg_v = model1.step()
            total_flow += avg_v * density
        flows1.append(total_flow / Te)
    
    # Случай 2: pslow=0.8, RHDV=0.7
    flows2 = []
    model2 = NaSchModel(Ncell, vmax, 0.8, 0.7)
    
    for density in densities:
        model2.initialize(density)
        for _ in range(Ts):
            model2.step()
        
        total_flow = 0
        for _ in range(Te):
            avg_v = model2.step()
            total_flow += avg_v * density
        flows2.append(total_flow / Te)
    
    # Построение на одном графике
    plt.scatter(densities, flows1, c='red', s=30, alpha=0.7, label='pslow=0.8, RHDV=0')
    plt.scatter(densities, flows2, c='blue', s=30, alpha=0.7, label='pslow=0.8, RHDV=0.7')
    
    plt.xlabel('Плотность [автомобилей/клетку]')
    plt.ylabel('Поток [автомобилей/шаг]')
    plt.title('Фундаментальные диаграммы модели NaSch')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return densities, flows1, flows2

def simulate_xt_diagram_NaSch():
    """x-t диаграммы для модели NaSch при плотности 0.3"""
    print("Модель NaSch: построение x-t диаграмм...")
    
    Ncell, Ts, Te, vmax = 1000, 2000, 500, 5
    density = 0.3
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Создаем собственную цветовую карту
    colors = ['red', 'orange', 'yellow', 'lime', 'green', 'darkgreen']
    cmap = plt.cm.colors.ListedColormap(colors)
    
    # Случай 1: pslow=0.8, RHDV=0
    print("Случай 1: pslow=0.8, RHDV=0")
    model = NaSchModel(Ncell, vmax, 0.8, 0)
    model.initialize(density)
    
    # Стабилизация
    for i in range(Ts):
        model.step()
    
    # Запись данных для x-t диаграммы (1000 шагов)
    xt_data1 = np.full((1000, Ncell), -1, dtype=float)
    speeds_list1 = []
    
    for t in range(1000):
        model.step()
        speeds_list1.extend(model.velocities.tolist())
        for i, (pos, vel) in enumerate(zip(model.positions, model.velocities)):
            xt_data1[t, pos] = vel
    
    # Маскируем пустые клетки (-1)
    masked_data1 = np.ma.masked_where(xt_data1 == -1, xt_data1)
    
    im1 = ax1.imshow(masked_data1, cmap=cmap, aspect='auto', 
                    extent=[0, Ncell, 1000, 0], vmin=0, vmax=vmax)
    ax1.set_xlabel('Позиция на дороге')
    ax1.set_ylabel('Время (шаги)')
    ax1.set_title('x-t: pslow=0.8, RHDV=0, ρ=0.3')
    plt.colorbar(im1, ax=ax1, label='Скорость')
    
    # Случай 2: pslow=0.8, RHDV=0.7
    print("Случай 2: pslow=0.8, RHDV=0.7")
    model = NaSchModel(Ncell, vmax, 0.8, 0.7)
    model.initialize(density)
    
    # Стабилизация
    for i in range(Ts):
        model.step()
    
    # Запись данных для x-t диаграммы (1000 шагов)
    xt_data2 = np.full((1000, Ncell), -1, dtype=float)
    speeds_list2 = []
    
    for t in range(1000):
        model.step()
        speeds_list2.extend(model.velocities.tolist())
        for i, (pos, vel) in enumerate(zip(model.positions, model.velocities)):
            xt_data2[t, pos] = vel
    
    # Маскируем пустые клетки (-1)
    masked_data2 = np.ma.masked_where(xt_data2 == -1, xt_data2)
    
    im2 = ax2.imshow(masked_data2, cmap=cmap, aspect='auto', 
                    extent=[0, Ncell, 1000, 0], vmin=0, vmax=vmax)
    ax2.set_xlabel('Позиция на дороге')
    ax2.set_ylabel('Время (шаги)')
    ax2.set_title('x-t: pslow=0.8, RHDV=0.7, ρ=0.3')
    plt.colorbar(im2, ax=ax2, label='Скорость')
    
    plt.tight_layout()
    plt.show()
    
    # Вывод статистики
    total_measurements1 = len(speeds_list1)
    total_measurements2 = len(speeds_list2)
    
    # Подсчет заполненности x-t диаграмм
    total_cells = 1000 * Ncell
    occupied_cells1 = np.sum(xt_data1 != -1)
    occupied_cells2 = np.sum(xt_data2 != -1)
    occupancy_rate1 = occupied_cells1 / total_cells
    occupancy_rate2 = occupied_cells2 / total_cells
    
    print(f"\nСтатистика заполненности x-t диаграмм:")
    print(f"Всего клеток на диаграмме: {total_cells}")
    print(f"Занято клеток (RHDV=0): {occupied_cells1} ({occupancy_rate1*100:.1f}%)")
    print(f"Занято клеток (RHDV=0.7): {occupied_cells2} ({occupancy_rate2*100:.1f}%)")
    print(f"Ожидаемая заполненность при ρ=0.3: {density*100:.1f}%")
    
    print(f"\nСтатистика скоростей:")
    print(f"Всего измерений (RHDV=0): {total_measurements1}")
    print(f"Всего измерений (RHDV=0.7): {total_measurements2}")
    
    for v in range(vmax + 1):
        count1 = sum(1 for speed in speeds_list1 if speed == v)
        count2 = sum(1 for speed in speeds_list2 if speed == v)
        percent1 = count1 / total_measurements1 * 100
        percent2 = count2 / total_measurements2 * 100
        print(f"Скорость {v}: RHDV=0 - {percent1:.1f}%, RHDV=0.7 - {percent2:.1f}%")
    
    avg_vel1 = np.mean(speeds_list1)
    avg_vel2 = np.mean(speeds_list2)
    print(f"\nСредняя скорость (RHDV=0): {avg_vel1:.2f}")
    print(f"Средняя скорость (RHDV=0.7): {avg_vel2:.2f}")

class S_NFS_Model:
    def __init__(self, L, vmax, p, q, r):
        self.L = L
        self.vmax = vmax
        self.p = p
        self.q = q
        self.r = r
        
    def initialize(self, density):
        self.N = int(density * self.L)
        if self.N == 0:
            self.N = 1
        self.positions = np.sort(random.sample(range(self.L), self.N))
        self.velocities = np.ones(self.N, dtype=int) * self.vmax
        
    def step(self):
        new_velocities = np.copy(self.velocities)
        new_positions = np.zeros(self.N, dtype=int)
        
        # Вычисление расстояний
        gaps = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if i < self.N - 1:
                gaps[i] = self.positions[i + 1] - self.positions[i] - 1
            else:
                gaps[i] = (self.L - self.positions[i] - 1) + self.positions[0]
        
        for i in range(self.N):
            v = self.velocities[i]
            gap = gaps[i]
            
            # Упрощенные правила S-NFS
            # 1. Ускорение
            if v < self.vmax:
                v += 1
            
            # 2. Торможение
            if v > gap:
                v = gap
            
            # 3. Медленный старт (с вероятностью q)
            if random.random() < self.q and v > 0:
                v -= 1
            
            # 4. Эффект упреждения (с вероятностью r)
            if random.random() < self.r and v < self.vmax and gap > v + 1:
                v += 1
            
            # 5. Случайное торможение
            if random.random() > self.p and v > 0:
                v -= 1
            
            new_velocities[i] = v
        
        # Движение
        for i in range(self.N):
            new_pos = (self.positions[i] + new_velocities[i]) % self.L
            new_positions[i] = new_pos
        
        sort_idx = np.argsort(new_positions)
        self.positions = new_positions[sort_idx]
        self.velocities = new_velocities[sort_idx]
        
        return np.mean(self.velocities)

def simulate_fundamental_diagram_SNFS():
    """Фундаментальные диаграммы для модели S-NFS на одном графике"""
    print("Модель S-NFS: построение фундаментальных диаграмм...")
    
    Ncell, Ts, Te, vmax, p = 1000, 200, 50, 3, 0.1
    # Генерируем 250 случайных плотностей от 0.01 до 1.0
    densities = np.random.rand(250) * 0.99 + 0.01
    densities = np.sort(densities)
    
    plt.figure(figsize=(10, 6))
    
    # Случай 1: q=1, r=0.5
    flows1 = []
    model1 = S_NFS_Model(Ncell, vmax, p, 1, 0.5)
    
    for density in densities:
        model1.initialize(density)
        for _ in range(Ts):
            model1.step()
        
        total_flow = 0
        for _ in range(Te):
            avg_v = model1.step()
            total_flow += avg_v * density
        flows1.append(total_flow / Te)
    
    # Случай 2: q=1, r=1
    flows2 = []
    model2 = S_NFS_Model(Ncell, vmax, p, 1, 1)
    
    for density in densities:
        model2.initialize(density)
        for _ in range(Ts):
            model2.step()
        
        total_flow = 0
        for _ in range(Te):
            avg_v = model2.step()
            total_flow += avg_v * density
        flows2.append(total_flow / Te)
    
    # Построение на одном графике
    plt.scatter(densities, flows1, c='green', s=30, alpha=0.7, label='q=1, r=0.5')
    plt.scatter(densities, flows2, c='purple', s=30, alpha=0.7, label='q=1, r=1')
    
    plt.xlabel('Плотность [автомобилей/клетку]')
    plt.ylabel('Поток [автомобилей/шаг]')
    plt.title('Фундаментальные диаграммы модели S-NFS')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return densities, flows1, flows2

# Основная программа
if __name__ == "__main__":
    print("=== Моделирование транспортных потоков ===")
    
    # 1. Модель NaSch
    print("\n1. Модель NaSch")
    simulate_fundamental_diagram_NaSch()
    simulate_xt_diagram_NaSch()
    
    # 2. Модель S-NFS
    print("\n2. Модель S-NFS")
    simulate_fundamental_diagram_SNFS()
    
    print("\nМоделирование завершено!")
