# Optimal Path Planning with Genetic Algorithm: A Visualization Approach for Obstacle-Avoidance  

## 📋 Overview  
This project demonstrates a **mobile pathfinding robot** utilizing a **Genetic Algorithm (GA)** for optimal path planning while avoiding obstacles. The project is part of a **Masters course in Bioinspired Systems**. It showcases the power of GA in solving complex optimization problems, emphasizing parameter tuning and visualization for improved performance.

---

## 🚀 Features  
- **Genetic Algorithm Implementation**:  
  - Value encoding for (x, y) coordinates.  
  - Path optimization with fixed start and goal points.  
  - Fitness function considers collision count, obstacle distance, path smoothness, and total path length.  

- **Adjustable Parameters**:  
  - **Mutation probability**: Controls diversity in the population.  
  - **Mutation range**: Defines variability during mutation.  
  - **Elitism rate**: Preserves top solutions.  

- **Selection and Crossover**:  
  - Top 50% retained through rank selection.  
  - Simple arithmetic crossover applied to middle genes.  

- **Visualization**:  
  - Interactive simulation using **Tkinter**.  
  - Real-time obstacle rendering and path updates.  

---

## 📊 Key Findings  
- **Parameter Optimization**:  
  - A mutation probability of **0.5** and a mutation range of **±0.75** yield the best results.  
  - Elitism rate shows minimal impact on performance.  

- **Convergence**:  
  - Optimal paths achieved within **50–400 generations**.  

- **Balanced Cost Weights**:  
  - A weighted sum of fitness factors ensures effective path optimization.  

---

## 🔧 Setup & Installation  

### Prerequisites  
- **Python 3.8+**  
- Libraries:  
  - `numpy`  
  - `matplotlib`  
  - `tkinter`  

### Installation  
1. Clone this repository:  
   ```bash  
   git clone https://github.com/malikfm/path-finding-robot.git  
   cd path-finding-robot  
   ```  

2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. Run the script:  
   ```bash  
   python main.py  
   ```  

---

## 🎮 Usage  
1. The simulation initializes with randomly placed obstacles.  
2. Watch the robot's path evolve toward optimal solutions over generations.  
3. View real-time updates of:  
   - Obstacles (squares, circles, triangles).  
   - Start and goal points.  
   - Best path visualization.  

---

## 📈 Results Visualization  
The algorithm's performance is tracked by plotting the **best fitness per generation**. This provides insights into the convergence speed and solution quality.  

---

## 🛡 License  
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.  
