#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <fstream>
#include <sstream>
using namespace std;
//namespace vecmat {

class Vector {
public:
	int size;
	double* vec;



	Vector(): size(1), vec(new double[1]){}

	Vector(int s) : size(s), vec(new double[s]) {
		for (int i = 0; i < s; i++)
			vec[i] = 0;
	}

	Vector(double* v, int s) : size(s), vec(v) {}



	double operator[](int i) {
		return vec[i];
	}

	Vector operator*(double num) {
		Vector res = Vector(size);
		for (int i = 0; i < size; i++)
			res.vec[i] = vec[i] * num;
		return res;
	}

	Vector operator*(Vector v2) {
		Vector res = Vector(size);
		if (size == v2.size) {
			for (int i = 0; i < size; i++)
				res.vec[i] = vec[i] * v2[i];
		}
		return res;
	}

	Vector operator+(Vector v2) {
		Vector res = Vector(size);
		if (size == v2.size) {
			for (int i = 0; i < size; i++)
				res.vec[i] = vec[i] + v2[i];
		}
		return res;
	}

	Vector operator-(Vector v2) {
		Vector res = Vector(size);
		if (size == v2.size) {
			for (int i = 0; i < size; i++)
				res.vec[i] = vec[i] - v2[i];
		}
		return res;
	}
	

	double* to_list() {
		return vec;
	}
};

class Matrix {
public:
	int size[2];
	double** mat;

	Matrix() : size{ 1, 1 }, mat(new double* [1]) {
		mat[0] = new double[1];
	}

	Matrix(Vector v):mat(new double*[1]) {
		mat[0] = v.vec;
		size[0] = 1;
		size[1] = v.size;
	}

	Matrix(int s_w, int s_h){
		size[0] = s_w;
		size[1] = s_h;
		mat = new double* [s_w];
		for (int i = 0; i < s_w; i++) {
			mat[i] = new double[s_h];
			for (int j = 0; j < s_h; j++)
				mat[i][j] = 0;
		}
	}

	Matrix(int s_w, int s_h, double** mat_) {
		size[0] = s_w;
		size[1] = s_h;
		mat = mat_;
	}

	Matrix(int s[2]) {
		size[0] = s[0];
		size[1] = s[1];
		mat = new double* [s[0]];
		for (int i = 0; i < s[0]; i++) {
			mat[i] = new double[s[1]];
			for (int j = 0; j < s[1]; j++)
				mat[i][j] = 0;
		}
	}

	Matrix operator*(Matrix m2) {
		Matrix m_res = Matrix(size[0], m2.size[1]);
		if (size[1] == m2.size[0]) {
			for (int i = 0; i < size[0]; i++) {
				for (int j = 0; j < m2.size[1]; j++) {
					double sum = 0;
					for (int n = 0; n < size[1]; ++n) {
						sum += mat[i][n] * m2.mat[n][j];
					}
					m_res.mat[i][j] = sum;
				}
			}
			return m_res;
		}
	}
	Matrix operator*(double num) {
		Matrix res = Matrix(size);
		for (int i = 0; i < size[0]; i++)
			for (int j = 0; j < size[1]; i++)
				res.mat[i][j] = mat[i][j] * num;
		return res;
	}
	Matrix operator-(Matrix m2) {
		Matrix res = Matrix(size);
		if ((size[0] == m2.size[0]) && (size[1] == m2.size[1])) {
			for (int i = 0; i < size[0]; i++)
				for (int j = 0; j < size[1]; i++)
					res.mat[i][j] = mat[i][j] - m2.mat[i][j];
		}
		return res;
	}

	Matrix T() {
		Matrix res = Matrix(size[1], size[0]);
		for (int i = 0; i < size[0]; i++) {
			for (int j = 0; j < size[1]; j++) {
				res.mat[j][i] = mat[i][j];
			}
		}
	}
};

class Perceptron {
public:
	double ALPHA = 0.00001;
	int* dims;
	int layer_count;
	int weight_count;
	Matrix* weights;
	Vector* biases;
	Vector* t;
	Vector* h;

	Perceptron(int* d, int d_count) : dims(new int[d_count]), layer_count(d_count), weight_count(d_count - 1) {
		for (int i = 0; i < d_count; i++) {
			dims[i] = d[i];
		}
		weights = new Matrix[weight_count];
		for (int i = 0; i < weight_count; i++) {
			weights[i] = rand_mat(dims[i], dims[i + 1]);
		}
		biases = new Vector[weight_count];
		for (int i = 0; i < weight_count; i++) {
			biases[i] = rand_vec(dims[i + 1]);
		}
		t = new Vector[weight_count];
		h = new Vector[weight_count];
	}

	Vector predict_class(Vector x) {
		h[0] = x;
		for (int i = 0; i < weight_count - 2; i++) {
			t[i] = Vector((Matrix(h[i]) * weights[i]).mat[0], weights[i].size[1]) + biases[i];
			h[i + 1] = relu(t[i]);
		}
		t[weight_count - 1] = softmax(h[weight_count - 1]);
		return t[weight_count - 1];
	}

	Vector predict_conv(Vector x) {
		h[0] = x;
		for (int i = 0; i < weight_count - 2; i++) {
			t[i] = Vector((Matrix(h[i]) * weights[i]).mat[0], weights[i].size[1]) + biases[i];
			h[i + 1] = relu(t[i]);
		}
		t[weight_count - 1] = relu(h[weight_count - 1]);
		return t[weight_count - 1];
	}

	Vector backprop_train(Vector dE_dz) {
		int d = weight_count;
		Vector* dE_dt = new Vector[d];
		Matrix* dE_dW = new Matrix[d];
		Vector* dE_db = new Vector[d];
		Vector* dE_dh = new Vector[d];

		dE_dt[d - 1] = dE_dz;
		dE_dW[d - 1] = Matrix(h[d - 1]).T() * Matrix(dE_dt[d - 1]);
		dE_db[d - 1] = dE_dt[d - 1];
		dE_dh[d - 1] = Vector((Matrix(dE_dt[d - 1]) * weights[d - 1].T()).mat[0], dims[d - 1]);

		for (int i = d - 2; i > -1; i--) {
			dE_dt[i] = dE_dh[i+1] * relu_deriv(t[i]);
			dE_dW[i] = Matrix(h[i]).T() * Matrix(dE_dt[i]);
			dE_db[i] = dE_dt[i];
			dE_dh[i] = Vector((Matrix(dE_dt[i]) * weights[i].T()).mat[0], dims[i]);
		}

		for (int i = 0; i < d; i++) {
			weights[i] = weights[i] - (dE_dW[i] * ALPHA);
			biases[i] = biases[i] - (dE_db[i] * ALPHA);
		}

		return dE_dh[0];

	}

	Vector backprop_train_conv(Vector dE_dz) {
		int d = weight_count;
		Vector* dE_dt = new Vector[d];
		Matrix* dE_dW = new Matrix[d];
		Vector* dE_db = new Vector[d];
		Vector* dE_dh = new Vector[d];

		dE_dt[d - 1] = dE_dz * relu_deriv(t[d - 1]);
		dE_dW[d - 1] = Matrix(h[d - 1]).T() * Matrix(dE_dt[d - 1]);
		dE_db[d - 1] = dE_dt[d - 1];
		dE_dh[d - 1] = Vector((Matrix(dE_dt[d - 1]) * weights[d - 1].T()).mat[0], dims[d - 1]);

		for (int i = d - 2; i > -1; i--) {
			dE_dt[i] = dE_dh[i + 1] * relu_deriv(t[i]);
			dE_dW[i] = Matrix(h[i]).T() * Matrix(dE_dt[i]);
			dE_db[i] = dE_dt[i];
			dE_dh[i] = Vector((Matrix(dE_dt[i]) * weights[i].T()).mat[0], dims[i]);
		}

		for (int i = 0; i < d; i++) {
			weights[i] = weights[i] - (dE_dW[i] * ALPHA);
			biases[i] = biases[i] - (dE_db[i] * ALPHA);
		}

		return dE_dh[0];
	}

private:
	double round_n(double integer, int n) {
		return (double)round(integer * pow(10, n)) / pow(10, n);
	}
	double random(int min, int max, int n = 6) {
		return round_n((double)(rand()) / RAND_MAX * (max - min), n);
	}
	Matrix rand_mat(int n, int m) {
		Matrix arr = Matrix(n, m);
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < m; ++j)
				arr.mat[i][j] = random(0, 1);
		return arr;
	}
	Vector rand_vec(int s) {
		Vector arr = Vector(s);
		for (int i = 0; i < s; ++i)
			arr.vec[i] = random(0, 1);
		return arr;
	}
	Vector relu(Vector x) {
		Vector out = Vector(x.size);
		for (int i = 0; i < x.size; i++) {
			if (x[i] >= 0.0) out.vec[i] = x[i];
			else out.vec[i] = 0;
		}
		return out;
	}
	Vector relu_deriv(Vector x) {
		Vector out = Vector(x.size);
		for (int i = 0; i < x.size; i++) {
			if (x[i] >= 0.0) out.vec[i] = 1;
			else out.vec[i] = 0;
		}
		return out;
	}
	Vector softmax(Vector t) {
		Vector out = Vector(t.size);
		double sum = 0.0;
		for (int i = 0; i < t.size; i++) {
			double x = exp(t[i]);
			out.vec[i] = x;
			sum += x;
		}
		for (int i = 0; i < t.size; i++) {
			out.vec[i] = out[i] / sum;
		}
		return out;
	}
	double cross_entropy(Vector t, Vector y) {
		double E = 0.0;
		for (int i = 0; i < t.size; i++) {
			E -= log((y * t)[i]);
		}
		return E;
	}
};

class ConvNet {
public:
	int conv_layer_count;
	int* classifier_DIMs;
	int classifier_DIMs_len;

	ConvNet(int c, int dl, int* d): conv_layer_count(c), classifier_DIMs_len(dl), classifier_DIMs(d){}

	Vector predict(Matrix input[3]) {

	}

private:
	struct MatDiff {
		Matrix out;
		Matrix diff;
	};
	MatDiff pooling(Matrix img) {
		int w = img.size[0] / 2;
		int h = img.size[1] / 2;
		MatDiff res;
		res.out = Matrix(w, h);
		res.diff = Matrix(img.size);
		for(int i = 0; i < w; i++)
			for (int j = 0; j < h; j++) {
				int index[2] = { 0, 0 };
				double max_ = 0.0;
				for(int x = 0; x < 2; x++)
					for (int y = 0; y < 2; y++) 
						if (img.mat[i * 2 + x][j * 2 + y] > max_) {
							max_ = img.mat[i * 2 + x][j * 2 + y];
							index[0] = x;
							index[1] = y;
						}
				res.out.mat[i][j] = max_;
				res.diff.mat[i * 2 + index[0]][j * 2 + index[1]] = 1;
			}
		return res;
	}
	Matrix padding(Matrix img) {
		int w = img.size[0] + 2;
		int h = img.size[1] + 2;
		Matrix res = Matrix(w, h);
		for(int i = 0; i < w; i++)
			for (int j = 0; j < h; j++) {
				int x, y;
				x = i;
				y = j;
				if (i == 0)
					x = 1;
				if (j == 0)
					y = 1;
				if (j == h - 1)
					y = h - 2;
				if (i == w - 1)
					x = w - 2;
				res.mat[i][j] = img.mat[x - 1][y - 1];
			}
		return res;
	}
	Matrix conv3(Perceptron kernel, Matrix img) {
		int w = img.size[0];
		int h = img.size[1];
		img = padding(img);
		Matrix res = Matrix(w, h);
		for(int i = 0; i < w; i++)
			for (int j = 0; j < h; j++) {
				Vector v = Vector(9);
				for (int x = 0; x < 3; x++)
					for (int y = 0; y < 3; y++)
						v.vec[x * 2 + y] = img.mat[i + x][j + y];
				res.mat[i][j] = kernel.predict_conv(v)[0];
			}
		return res;
	}
};
//}
int main()
{
	int dl_[4] = { 16, 8, 4, 2 };
	int* dlp = new int[4];
	for (int i = 0; i < 4; i++)
		dlp[i] = dl_[i];
	ConvNet neuro = ConvNet(2, 3, dlp);
	
}







/*
#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <fstream>
#include <sstream>
using namespace std;

class Matrix {
public:
	int m_w;
	int m_h;
	double** mat;

	Matrix() : m_w(1), m_h(1) {}
	Matrix(int w, int h) :
		m_w(w), m_h(h)
	{
		mat = new double* [m_w];
		for (int i = 0; i < m_w; i++) {
			mat[i] = new double[m_h];
			for (int j = 0; j < m_h; j++) {
				mat[i][j] = 0.0;
			}
		}
	}
	Matrix(double** matrix, int w, int h) :
		m_w(w), m_h(h), mat(matrix)
	{
	}
	Matrix(double* v, int v_s) : m_w(1), m_h(v_s), mat(new double* [1])
	{
		mat[0] = v;
	}

	Matrix operator*(Matrix& other) {
		if (m_h == other.m_w) {
			Matrix m_res = Matrix(m_w, other.m_h);
			for (int i = 0; i < m_w; i++) {
				for (int j = 0; j < other.m_h; j++) {
					double sum = 0;
					for (int n = 0; n < m_h; ++n) {
						sum += mat[i][n] * other.mat[n][j];
					}
					m_res.mat[i][j] = sum;
				}
			}
			return m_res;
		}
		else {
			throw MatrixMultException(other.m_w, other.m_h);
		}
	}
	Matrix operator+(Matrix& other) {
		if ((m_w == other.m_w) && (m_h == other.m_h)) {
			Matrix m_res = Matrix(m_w, m_h);
			for (int i = 0; i < m_w; i++) {
				for (int j = 0; j < m_h; j++) {
					m_res.mat[i][j] = mat[i][j] + other.mat[i][j];
				}
			}
			return m_res;
		}
		else {
			throw MatrixSumException(other.m_w, other.m_h);
		}
	}
	Matrix operator-(Matrix& other) {
		if ((m_w == other.m_w) && (m_h == other.m_h)) {
			Matrix m_res = Matrix(m_w, m_h);
			for (int i = 0; i < m_w; i++) {
				for (int j = 0; j < m_h; j++) {
					m_res.mat[i][j] = mat[i][j] - other.mat[i][j];
				}
			}
			return m_res;
		}
		else {
			throw MatrixSubException(other.m_w, other.m_h);
		}
	}

	Matrix T() {
		Matrix res = Matrix(m_h, m_w);
		for (int i = 0; i < m_w; i++) {
			for (int j = 0; j < m_h; j++) {
				res.mat[j][i] = mat[i][j];
			}
		}
	}
private:
	std::exception MatrixMultException(int other_w, int other_h) {
		std::exception err("Incorrect matrix multiplying! DIMs should be: {N * M}, {M * K}");
		cout << "Incorrect matrix multiplying! DIMs should be: {N * M}, {M * K}" << std::endl;
		cout << "Matrix DIMs: {" << m_w << " * " << m_h << "}, {" << other_w << " * " << other_h << "}" << std::endl;
		return err;
	}
	std::exception MatrixSumException(int other_w, int other_h) {
		std::exception err("Incorrect matrix addition! DIMs should be equal : {N* M}, {N * M}");
		cout << "Incorrect matrix addition! DIMs should be equal: {N* M}, {N * M}" << std::endl;
		cout << "Matrix DIMs: {" << m_w << " * " << m_h << "}, {" << other_w << " * " << other_h << "}" << std::endl;
		return err;
	}
	std::exception MatrixSubException(int other_w, int other_h) {
		std::exception err("Incorrect matrix substraction! DIMs should be equal : {N* M}, {N * M}");
		cout << "Incorrect matrix substraction! DIMs should be equal: {N* M}, {N * M}" << std::endl;
		cout << "Matrix DIMs: {" << m_w << " * " << m_h << "}, {" << other_w << " * " << other_h << "}" << std::endl;
		return err;
	}
};

class Vector {
public:
	int size;
	double* vec;

	Vector() : size(1), vec(new double[1]) {}
	Vector(int s) :
		size(s)
	{
		vec = new double[size];
		for (int i = 0; i < size; i++) {
			vec[i] = 0.0;
		}
	}
	Vector(double* vector, int s) :
		size(s), vec(vector)
	{
	}

	Vector operator*(Vector other) {
		if (size == other.size) {
			Vector v_res = Vector(size);
			for (int i = 0; i < size; i++) {
				v_res.vec[i] = vec[i] * other.vec[i];
			}
			return v_res;
		}
		else {
			throw VectorMultException(other.size);
		}
	}
	Vector operator+(Vector other) {
		if (size == other.size) {
			Vector v_res = Vector(size);
			for (int i = 0; i < size; i++) {
				v_res.vec[i] = vec[i] + other.vec[i];
			}
			return v_res;
		}
		else {
			throw VectorSumException(other.size);
		}
	}
	Vector operator-(Vector other) {
		if (size == other.size) {
			Vector v_res = Vector(size);
			for (int i = 0; i < size; i++) {
				v_res.vec[i] = vec[i] - other.vec[i];
			}
			return v_res;
		}
		else {
			throw VectorSubException(other.size);
		}
	}

	Matrix T() {
		Matrix res = Matrix(size, 1);
		for (int i = 0; i < size; i++) {
			res.mat[i][0] = vec[i];
		}
		return res;
	}
private:
	std::exception VectorMultException(int other_s) {
		std::exception err("Incorrect vector multiplying! DIMs should be: {N}, {N}");
		cout << "Incorrect vector multiplying! DIMs should be: {N}, {N}" << std::endl;
		cout << "Vector DIMs: {" << size << "}, {" << other_s << "}" << std::endl;
		return err;
	}
	std::exception VectorSumException(int other_s) {
		std::exception err("Incorrect vector addition! DIMs should be: {N}, {N}");
		cout << "Incorrect vector addition! DIMs should be: {N}, {N}" << std::endl;
		cout << "Vector DIMs: {" << size << "}, {" << other_s << "}" << std::endl;
		return err;
	}
	std::exception VectorSubException(int other_s) {
		std::exception err("Incorrect vector substraction! DIMs should be: {N}, {N}");
		cout << "Incorrect vector substraction! DIMs should be: {N}, {N}" << std::endl;
		cout << "Vector DIMs: {" << size << "}, {" << other_s << "}" << std::endl;
		return err;
	}
};

class Perceptron {
public:
	int* dims;
	int layer_count;
	int weight_count;
	Matrix* weights;
	Vector* biases;
	Vector* t;
	Vector* h;

	Perceptron(int* d, int d_count) : dims(new int[d_count]), layer_count(d_count), weight_count(d_count - 1) {
		for (int i = 0; i < d_count; i++) {
			dims[i] = d[i];
		}
		weights = new Matrix[weight_count];
		for (int i = 0; i < weight_count; i++) {
			weights[i] = rand_mat(dims[i], dims[i + 1]);
		}
		biases = new Vector[weight_count];
		for (int i = 0; i < weight_count; i++) {
			biases[i] = rand_vec(dims[i + 1]);
		}
		t = new Vector[weight_count];
		h = new Vector[weight_count];
	}

	Vector predict(Vector x) {
		h[0] = x;
		for (int i = 0; i < weight_count - 2; i++) {
			double** s = new double* [1];
			s[0] = h[i].vec;
			t[i] = Vector((Matrix(s, 1, h[i].size) * weights[i]).mat[0], weights[i].m_h) + biases[i];
			h[i + 1] = relu(t[i]);
		}
		t[weight_count - 1] = softmax(h[weight_count - 1]);
		return h[weight_count - 1];
	}

	void backprop_train(Vector dE_dz) {
		int d = weight_count;
		Vector* dE_dt = new Vector[d];
		Vector* dE_dW = new Vector[d];
		Vector* dE_db = new Vector[d];
		Vector* dE_dh = new Vector[d - 1];

		dE_dt[d - 1] = dE_dz * relu_deriv(t[d - 1]);

	}

private:
	double round_n(double integer, int n) {
		return (double)round(integer * pow(10, n)) / pow(10, n);
	}
	double random(int min, int max, int n = 6) {
		return round_n((double)(rand()) / RAND_MAX * (max - min), n);
	}
	Matrix rand_mat(int n, int m) {
		Matrix arr = Matrix(n, m);
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < m; ++j)
				arr.mat[i][j] = random(0, 1);
		return arr;
	}
	Vector rand_vec(int s) {
		Vector arr = Vector(s);
		for (int i = 0; i < s; ++i)
			arr.vec[i] = random(0, 1);
		return arr;
	}
	Vector relu(Vector x) {
		Vector out = Vector(x.size);
		for (int i = 0; i < x.size; i++) {
			if (x.vec[i] >= 0.0) out.vec[i] = x.vec[i];
			else out.vec[i] = 0;
		}
		return out;
	}
	Vector relu_deriv(Vector x) {
		Vector out = Vector(x.size);
		for (int i = 0; i < x.size; i++) {
			if (x.vec[i] >= 0.0) out.vec[i] = 1;
			else out.vec[i] = 0;
		}
		return out;
	}
	Vector softmax(Vector t) {
		Vector out = Vector(t.size);
		double sum = 0.0;
		for (int i = 0; i < t.size; i++) {
			double x = exp(t.vec[i]);
			out.vec[i] = x;
			sum += x;
		}
		for (int i = 0; i < t.size; i++) {
			out.vec[i] = out.vec[i] / sum;
		}
		return out;
	}
	double cross_entropy(Vector t, Vector y) {
		double E = 0.0;
		for (int i = 0; i < t.size; i++) {
			E -= log(y.vec[i] * t.vec[i]);
		}
		return E;
	}
};

int main()
{
	int* d = new int[3];
	d[0] = 4;
	d[1] = 5;
	d[2] = 3;
	Perceptron nn = Perceptron(d, 3);
	Vector test = Vector(4);
	Vector out = nn.predict(test);
}
*/
// Запуск программы: CTRL+F5 или меню "Отладка" > "Запуск без отладки"
// Отладка программы: F5 или меню "Отладка" > "Запустить отладку"

// Советы по началу работы 
//   1. В окне обозревателя решений можно добавлять файлы и управлять ими.
//   2. В окне Team Explorer можно подключиться к системе управления версиями.
//   3. В окне "Выходные данные" можно просматривать выходные данные сборки и другие сообщения.
//   4. В окне "Список ошибок" можно просматривать ошибки.
//   5. Последовательно выберите пункты меню "Проект" > "Добавить новый элемент", чтобы создать файлы кода, или "Проект" > "Добавить существующий элемент", чтобы добавить в проект существующие файлы кода.
//   6. Чтобы снова открыть этот проект позже, выберите пункты меню "Файл" > "Открыть" > "Проект" и выберите SLN-файл.
