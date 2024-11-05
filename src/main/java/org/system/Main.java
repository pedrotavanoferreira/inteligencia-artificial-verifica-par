package org.system;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class Main {
    public static void main(String[] args) {
        // Create a single attribute for the number
        Attribute attribute = new Attribute("number");

        // Create a FastVector for class labels (even/odd)
        FastVector values = new FastVector(2);
        values.addElement("even");
        values.addElement("odd");
        Attribute classAttribute = new Attribute("class", values);

        // Create an empty Instances object
        Instances data = new Instances("ParImpar", new FastVector(1), 0);
        data.setAttributeWeight(attribute, 0);
        data.setClassIndex(data.numAttributes() - 1);

        // Create a new instance for classification (replace 42 with your number)
        double[] instanceValues = {42};
        Instance instance = new Instance(1.0, instanceValues);
        data.add(instance);

        // Build a Naive Bayes classifier
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(data);

        // Classify the instance
        double cls = nb.classifyInstance(instance);
        String[] classLabels = data.classAttribute().enumerateValues();
        System.out.println(classLabels[(int) cls]);

    }
}