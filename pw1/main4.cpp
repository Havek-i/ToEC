#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <thread>
#include <array>
#include <omp.h>

using namespace std;

// ==========================================
// ПЕРЕЧИСЛЕНИЯ (ENUM CLASS)
// ==========================================

enum class SelectionType { 
    RouletteWheel, 
    Rank, 
    Tournament 
};

enum class Crossing { 
    SinglePoint, 
    TwoPoint, 
    Uniform 
};

enum class MutationType { 
    Weak, 
    Average, 
    Strong 
};

enum class FormationGenerationType { 
    Childs, 
    ChildsAndParent, 
    TheBestChildsAndParents 
};

// Массивы для удобного перебора в циклах
constexpr array<SelectionType, 3> ALL_SELECTIONS = {
    SelectionType::RouletteWheel, SelectionType::Rank, SelectionType::Tournament
};
constexpr array<Crossing, 3> ALL_CROSSINGS = {
    Crossing::SinglePoint, Crossing::TwoPoint, Crossing::Uniform
};
constexpr array<MutationType, 3> ALL_MUTATIONS = {
    MutationType::Weak, MutationType::Average, MutationType::Strong
};
constexpr array<FormationGenerationType, 3> ALL_REPLACEMENTS = {
    FormationGenerationType::Childs, FormationGenerationType::ChildsAndParent, FormationGenerationType::TheBestChildsAndParents
};

// Функции для преобразования enum в строку (для CSV)
string toString(SelectionType v) {
    switch (v) {
        case SelectionType::RouletteWheel: return "RouletteWheel";
        case SelectionType::Rank: return "Rank";
        case SelectionType::Tournament: return "Tournament";
    }
    return "Unknown";
}
string toString(Crossing v) {
    switch (v) {
        case Crossing::SinglePoint: return "SinglePoint";
        case Crossing::TwoPoint: return "TwoPoint";
        case Crossing::Uniform: return "Uniform";
    }
    return "Unknown";
}
string toString(MutationType v) {
    switch (v) {
        case MutationType::Weak: return "Weak";
        case MutationType::Average: return "Average";
        case MutationType::Strong: return "Strong";
    }
    return "Unknown";
}
string toString(FormationGenerationType v) {
    switch (v) {
        case FormationGenerationType::Childs: return "Childs";
        case FormationGenerationType::ChildsAndParent: return "ChildsAndParent";
        case FormationGenerationType::TheBestChildsAndParents: return "TheBestChildsAndParents";
    }
    return "Unknown";
}

// ==========================================
// ГЕНЕРАТОР СЛУЧАЙНЫХ ЧИСЕЛ
// ==========================================
thread_local mt19937 rng(random_device{}() ^ hash<thread::id>{}(this_thread::get_id()));

int random_int(int min, int max) {
    uniform_int_distribution<int> dist(min, max);
    return dist(rng);
}
double random_double(double min, double max) {
    uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

// ==========================================
// ОПИСАНИЕ ЗАДАЧ (1-12)
// ==========================================
int eval_task1(const vector<uint8_t>& x, const vector<uint8_t>& mask, int L) {
    int sum = 0; for (int i = 0; i < L; ++i) sum += x[i]; return sum;
}
int eval_task2(const vector<uint8_t>& x, const vector<uint8_t>& mask, int L) {
    int sum = 0; for (int i = 0; i < L; ++i) sum += (i + 1) * x[i]; return sum;
}
int eval_task3(const vector<uint8_t>& x, const vector<uint8_t>& mask, int L) {
    int sum = 0; for (int i = 0; i < L; ++i) { if (x[i] == 1) sum++; else break; } return sum;
}
int eval_task4(const vector<uint8_t>& x, const vector<uint8_t>& mask, int L) {
    int sum = 0; for (int i = 0; i < L; ++i) sum += mask[i] * x[i]; return sum;
}
int eval_task5(const vector<uint8_t>& x, const vector<uint8_t>& mask, int L) {
    int sum = 0; for (int i = 0; i < L; ++i) sum += mask[i] * (i + 1) * x[i]; return sum;
}
int eval_task6(const vector<uint8_t>& x, const vector<uint8_t>& mask, int L) {
    int sum = 0; int prod = 1;
    for (int i = 0; i < L; ++i) { prod *= x[i]; sum += mask[i] * prod; } return sum;
}
int eval_task7(const vector<uint8_t>& x, const vector<uint8_t>& mask, int L) {
    int sum = 0; int k = L / 3;
    for (int j = 0; j < k; ++j) { if (x[3*j] + x[3*j+1] + x[3*j+2] >= 2) sum++; } return sum;
}
int eval_task8(const vector<uint8_t>& x, const vector<uint8_t>& mask, int L) {
    int sum = 0; int k = L / 3;
    for (int j = 0; j < k; ++j) { if (x[3*j] + x[3*j+1] + x[3*j+2] >= 2) sum += (j + 1); } return sum;
}
int eval_task9(const vector<uint8_t>& x, const vector<uint8_t>& mask, int L) { return 0; }
int apply_ruggedness(int v, int s, int L) {
    if (v == s) return ceil(L / 2.0) + 1;
    if (v % 2 == 0) return floor(v / 2.0) + 1;
    return ceil(v / 2.0) + 1;
}
int eval_task10(const vector<uint8_t>& x, const vector<uint8_t>& mask, int L) {
    return apply_ruggedness(eval_task1(x, mask, L), L, L);
}
int eval_task11(const vector<uint8_t>& x, const vector<uint8_t>& mask, int L) {
    return apply_ruggedness(eval_task2(x, mask, L), L * (L + 1) / 2, L);
}
int eval_task12(const vector<uint8_t>& x, const vector<uint8_t>& mask, int L) {
    return apply_ruggedness(eval_task3(x, mask, L), L, L);
}

typedef int (*EvalFunc)(const vector<uint8_t>&, const vector<uint8_t>&, int);
EvalFunc tasks[] = {
    eval_task1, eval_task2, eval_task3, eval_task4,
    eval_task5, eval_task6, eval_task7, eval_task8,
    eval_task9, eval_task10, eval_task11, eval_task12
};

// ==========================================
// СТРУКТУРА КОНФИГУРАЦИИ И ГЕНЕТИЧЕСКИЙ АЛГОРИТМ
// ==========================================

struct Individual {
    vector<uint8_t> genome;
    double fitness;
};

struct GAConfig {
    int L, N, G;
    SelectionType sel;
    Crossing cross;
    MutationType mut;
    FormationGenerationType repl;
};

class GeneticAlgorithm {
private:
    GAConfig cfg;
    EvalFunc eval_func;
    vector<uint8_t> mask;
    vector<Individual> pop;

int select_parent() {
        switch (cfg.sel) {
            case SelectionType::RouletteWheel: {
                double sum_fit = 0;
                for (const auto& ind : pop) sum_fit += ind.fitness;
                if (sum_fit == 0) return random_int(0, cfg.N - 1);
                
                double r = random_double(0, sum_fit);
                double current = 0;
                for (int i = 0; i < cfg.N; ++i) {
                    current += pop[i].fitness;
                    if (current >= r) return i;
                }
                return cfg.N - 1;
            }
            
            case SelectionType::Rank: {
                vector<int> indices(cfg.N);
                iota(indices.begin(), indices.end(), 0);
                sort(indices.begin(), indices.end(), [&](int a, int b) {
                    return pop[a].fitness > pop[b].fitness;
                });
                
                double sum_ranks = cfg.N * (cfg.N + 1) / 2.0;
                double r = random_double(0, sum_ranks);
                double current = 0;
                for (int i = 0; i < cfg.N; ++i) {
                    current += (cfg.N - i);
                    if (current >= r) return indices[i];
                }
                return indices[cfg.N - 1];
            }
            
            case SelectionType::Tournament: {
                int p1 = random_int(0, cfg.N - 1);
                int p2 = random_int(0, cfg.N - 1);
                return (pop[p1].fitness > pop[p2].fitness) ? p1 : p2;
            }
        }
        
        // Недостижимый код, нужен для успокоения компилятора
        return 0; 
    }

    void crossover(const vector<uint8_t>& p1, const vector<uint8_t>& p2, vector<uint8_t>& c1, vector<uint8_t>& c2) {
        c1 = p1; 
        c2 = p2;
        
        switch (cfg.cross) {
            case Crossing::SinglePoint: {
                int cut = random_int(1, cfg.L - 1);
                for (int i = cut; i < cfg.L; ++i) {
                    swap(c1[i], c2[i]);
                }
                break; // Обязательно прерываем выполнение
            }
            
            case Crossing::TwoPoint: {
                int cut1 = random_int(1, cfg.L - 2);
                int cut2 = random_int(cut1 + 1, cfg.L - 1);
                for (int i = cut1; i < cut2; ++i) {
                    swap(c1[i], c2[i]);
                }
                break;
            }
            
            case Crossing::Uniform: {
                for (int i = 0; i < cfg.L; ++i) {
                    if (random_int(0, 1) == 1) {
                        swap(c1[i], c2[i]);
                    }
                }
                break;
            }
        }
    }

    void mutate(vector<uint8_t>& x) {
        double pm = 0.0;
        
        switch (cfg.mut) {
            case MutationType::Weak: {
                pm = 1.0 / (3.0 * cfg.L);
                break;
            }
            case MutationType::Average: {
                pm = 1.0 / cfg.L;
                break;
            }
            case MutationType::Strong: {
                pm = min(3.0 / cfg.L, 1.0);
                break;
            }
        }
        
        // Применяем мутацию с рассчитанной вероятностью pm
        for (int i = 0; i < cfg.L; ++i) {
            if (random_double(0, 1) < pm) {
                x[i] = 1 - x[i];
            }
        }
    }

public:
    GeneticAlgorithm(const GAConfig& config, EvalFunc ef, const vector<uint8_t>& m)
        : cfg(config), eval_func(ef), mask(m) {}

    double run() {
        pop.resize(cfg.N);
        for (int i = 0; i < cfg.N; ++i) {
            pop[i].genome.resize(cfg.L);
            for (int j = 0; j < cfg.L; ++j) pop[i].genome[j] = random_int(0, 1);
            pop[i].fitness = eval_func(pop[i].genome, mask, cfg.L);
        }

        for (int gen = 0; gen < cfg.G; ++gen) {
            vector<Individual> offspring(cfg.N);
            
            for (int i = 0; i < cfg.N / 2; ++i) {
                int p1_idx = select_parent();
                int p2_idx = select_parent();
                
                vector<uint8_t> c1, c2;
                crossover(pop[p1_idx].genome, pop[p2_idx].genome, c1, c2);
                
                mutate(c1); mutate(c2);
                
                offspring[2*i].genome = c1;
                offspring[2*i].fitness = eval_func(c1, mask, cfg.L);
                offspring[2*i+1].genome = c2;
                offspring[2*i+1].fitness = eval_func(c2, mask, cfg.L);
            }
            
            if (cfg.repl == FormationGenerationType::Childs) {
                pop = offspring;
            } 
            else if (cfg.repl == FormationGenerationType::ChildsAndParent) {
                int best_idx = 0;
                for (int i = 1; i < cfg.N; ++i) if (pop[i].fitness > pop[best_idx].fitness) best_idx = i;
                
                int worst_off_idx = 0;
                for (int i = 1; i < cfg.N; ++i) if (offspring[i].fitness < offspring[worst_off_idx].fitness) worst_off_idx = i;
                
                if (pop[best_idx].fitness > offspring[worst_off_idx].fitness) offspring[worst_off_idx] = pop[best_idx];
                pop = offspring;
            } 
            else { // TheBestChildsAndParents
                vector<Individual> combined;
                combined.reserve(2 * cfg.N);
                for (const auto& ind : pop) combined.push_back(ind);
                for (const auto& ind : offspring) combined.push_back(ind);
                
                sort(combined.begin(), combined.end(), [](const Individual& a, const Individual& b) {
                    return a.fitness > b.fitness;
                });
                
                for (int i = 0; i < cfg.N; ++i) pop[i] = combined[i];
            }
        }

        double best_fit = 0;
        for (const auto& ind : pop) if (ind.fitness > best_fit) best_fit = ind.fitness;
        return best_fit;
    }
};

// ==========================================
// MAIN (ТЕСТИРОВАНИЕ)
// ==========================================

int main() {
    cout << std::unitbuf; // Отключаем буферизацию консоли
    cout << "Available CPU cores: " << omp_get_max_threads() << endl;

    ofstream out("results.csv");
    if (!out.is_open()) {
        cerr << "ERROR: Cannot open results.csv for writing!" << endl;
        return 1;
    }
    out << std::unitbuf; // Отключаем буферизацию файла (запись каждой строки сразу на диск)
    
    // Заголовок CSV с понятными названиями
    out << "Task,Length,SelectionType,CrossingType,MutationType,FormationGenerationType,Mean,StdDev\n";

    int R = 100;  // Количество прогонов
    int N = 100;  // Размер популяции
    int G = 100;  // Количество поколений

    vector<int> lengths = {128, 512};
    int num_tasks = 12;

    cout << "Starting GA benchmark... Total runs: " << (2 * num_tasks * 81 * R) << endl;

    for (int L : lengths) {
        for (int task = 0; task < num_tasks; ++task) {
            // Создание маски для задач с Dummy Variables (4, 5, 6)
            vector<uint8_t> mask(L, 1);
            if (task == 3 || task == 4 || task == 5) {
                for (int i = 0; i < L; ++i) mask[i] = (random_double(0, 1) <= 0.8) ? 1 : 0;
            }

            // Перебор всех комбинаций с использованием enum
            for (SelectionType sel : ALL_SELECTIONS) {
                for (Crossing cross : ALL_CROSSINGS) {
                    for (MutationType mut : ALL_MUTATIONS) {
                        for (FormationGenerationType repl : ALL_REPLACEMENTS) {
                            
                            // Прогресс в консоли
                            cout << ">> Task " << (task + 1) << " | L=" << L 
                                 << " | Sel: " << toString(sel) 
                                 << " | Cross: " << toString(cross) 
                                 << " | Mut: " << toString(mut) 
                                 << " | Repl: " << toString(repl) << endl;

                            GAConfig config = { L, N, G, sel, cross, mut, repl };
                            vector<double> run_results(R);
                            
                            // Параллельное выполнение прогонов
                            #pragma omp parallel for schedule(dynamic)
                            for (int r = 0; r < R; ++r) {
                                GeneticAlgorithm ga(config, tasks[task], mask);
                                run_results[r] = ga.run();
                            }

                            // Расчет мат. ожидания и СКО
                            double sum = 0;
                            for (double v : run_results) sum += v;
                            double mean = sum / R; 

                            double sq_sum = 0;
                            for (double v : run_results) sq_sum += (v - mean) * (v - mean);
                            double stddev = sqrt(sq_sum / R);

                            // Запись результатов с понятными названиями
                            out << (task + 1) << "," << L << "," 
                                << toString(sel) << "," 
                                << toString(cross) << "," 
                                << toString(mut) << "," 
                                << toString(repl) << "," 
                                << fixed << setprecision(4) << mean << "," << stddev << "\n";
                        }
                    }
                }
            }
        }
    }

    out.close();
    cout << "Testing complete. Results saved to results.csv" << endl;
    return 0;
}


// Группировка данных по задачам длине строки, по [селекции, мутация...], чтобы посмотреть какое влияние оказывает на итоговую пригодность конкретный вид [...]
// Можно построить графики (столбчатые диаграммы) с использованием 