package org.system;
import weka.classifiers.trees.J48; // Utilizando um classificador J48 (árvore de decisão)
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ClassificadorParImpar {
    public static void main(String[] args) throws Exception {
        // Carrega o conjunto de dados a partir do arquivo ARFF
        DataSource source = new DataSource("data/data.arff");
        Instances data = source.getDataSet();

        // Define o índice do atributo classe (último atributo por padrão)
        data.setClassIndex(data.numAttributes() - 1);

        // Cria um classificador J48
        J48 j48 = new J48();

        // Treina o classificador com os dados
        j48.buildClassifier(data);

        // Cria uma nova instância para classificar (exemplo: número 42)
        double[] newInst = {42};
        Instances unlabeled = new Instances(data, 1);
        unlabeled.deleteAttributeAt(0); // Remove o atributo de classe (já que estamos prevendo)
        unlabeled.add(new Instance(1.0, newInst));

        // Classifica a instância
        double cls = j48.classifyInstance(unlabeled.instance(0));
        String[] classLabels = data.classAttribute().enumerateValues();
        System.out.println("O número " + newInst[0] + " é " + classLabels[(int) cls]+".");
    }
}