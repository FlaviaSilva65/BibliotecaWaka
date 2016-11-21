package iasemana_13;

import java.io.FileReader;
import java.util.Random;

import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

public class semana13 {

	// PARÃ‚METROS PARA A VALIDAÃ‡ÃƒO CRUZADA
	public static int PARTICOES = 3;
	public static int ITERACOES = 1;

	public static void main(String[] args) throws Exception {

		FileReader leitor = new FileReader("iris.arff");
		Instances iris = new Instances(leitor);
		iris.setClassIndex(4);
		iris = iris.resample(new Random());

		for (int i = 0; i < ITERACOES; i++) {

			// Obtendo as partiÃ§Ãµes de treino e de teste
			Instances irisTreino = iris.trainCV(PARTICOES, i);
			Instances irisTeste = iris.testCV(PARTICOES, i);

			// Definindo os classificadores
			IBk vizinho = new IBk();
			IBk knn = new IBk(3);

			// Treinando os classificadores
			vizinho.buildClassifier(irisTreino);
			knn.buildClassifier(irisTreino);

			// Anotando os resultados - SaÃ­da para um arquivo csv
			System.out.println("Real;Vizinho;kNN(3)");
			for (int j = 0; j < irisTeste.numInstances(); j++) {

				// Obtendo o exemplo que a ser classificado
				Instance exemplo = irisTeste.instance(j);

				System.out.print(exemplo.value(4)); // classe real
				exemplo.setClassMissing(); // removendo a classe

				double vizinhoRes = vizinho.classifyInstance(exemplo);
				double knnRes = knn.classifyInstance(exemplo);

				// respostas dos classificadores avaliados
				System.out.println(";" + vizinhoRes + ";" + knnRes);
			}
		}

	}

}

    

