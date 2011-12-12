package org.apache.giraph.benchmark;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.PosixParser;
import org.apache.giraph.graph.BasicVertex;
import org.apache.giraph.graph.BspUtils;
import org.apache.giraph.graph.GiraphJob;
import org.apache.giraph.graph.Vertex;
import org.apache.giraph.graph.VertexReader;
import org.apache.giraph.graph.VertexWriter;
import org.apache.giraph.lib.TextVertexInputFormat;
import org.apache.giraph.lib.TextVertexOutputFormat;
import org.apache.giraph.lib.TextVertexInputFormat.TextVertexReader;
import org.apache.giraph.lib.TextVertexOutputFormat.TextVertexWriter;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.BooleanWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import com.google.common.collect.Maps;

/**
 * DecisionTreeNodeVertex - Giraph Vertex for Decision Tree Classification
 * @author genia
 * 
 * Implementation based on the ID3 decision tree algorithm. Each vertex represents
 * a node in the decision tree. The root node (identified as 0) propogates training
 * and testing data through the tree.
 * 
 * Internal Node Design
 * Adapting a decision tree to graphical form is fairly intuitive - the key
 * criteria is that every node must be able to perform the same operations as
 * the root node, which include splitting the attribute set by maximum
 * information gain, creating new leaf nodes, and classifying a test set
 * example. 
 * 
 * Phase 1: Training [Building Decision Tree]
 * A single node-tree is the simplest decision tree. Vertices are added to the
 * graph as the tree grows, and they in turn generate their own child vertices
 * or terminate as leaves. The ID of the rootnode is propogated through messages to
 * child nodes.
 * 
 * Phase 2: Testing [Classification]
 * A single test datapoint is passed to a node, and is then passed as a message
 * to either the appropriate child node or as a classification back to the root
 * node. This essentially creates a classification queue at the root node. 
 * 
 * Giraph Vertex Design
 * @param <I> Vertex Id
 * @param <V> Vertex Value
 * @param <E> Edge Value
 * @param <M> Message Value
 */
public class DecisionTreeNodeVertex extends
Vertex<LongWritable, Text, FloatWritable, MapWritable>
implements Tool {

	// Quantity of training cases
	public static String TRAINING_DATA_CASES = "DecisionTreeNodeVertex.trainingDataCases";
	
	// Position of target attribute in input arrays
	public static String TARGET_ATTRIBUTE_POSITION = "DecisionTreeNodeVertex.targetAttributePosition";
	
	private MapWritable classificationData;
	
	public void setData(MapWritable data) {
		this.classificationData = data;
	}
	
	public MapWritable getData() {
		return classificationData;
	}
	
	/* Begin DecisionTreeNode behavior */

	/**
	 * addChildNode
	 * 
	 * Adds a destination vertex for an attribute value decision
	 * 
	 * @param FloatWritable decision
	 * @param LongWritable destination
	 */
	public void addChildNode (FloatWritable decision, LongWritable destination) {

		((MapWritable)classificationData.get("children")).put(decision, destination);
	
	}
	
	/**
	 * setSplitAttributeKey
	 * 
	 * Sets the split attribute key of this DecisionTreeNode's split
	 * 
	 * @param attributeKey
	 */
	public void setSplitAttributeKey (IntWritable attributeKey) {
		
		classificationData.put(new Text("classifyBy"), attributeKey);
	
	}
	
	/**
	 * excludeSplitAttributeKey
	 * 
	 * attributes without main split key included
	 * 
	 * @param attributes
	 * @return attributes - split
	 */
	public ArrayList<IntWritable> excludeSplitAttributeKey(ArrayList<IntWritable> attributes) {
		
		// Get split key
		IntWritable split = (IntWritable) classificationData.get(new Text("classifyBy"));
		
		// Return attribute list without split key
		attributes.remove(attributes.indexOf(split));
		
		return attributes;
	}
	
	/**
	 * growDecisionTree
	 * 
	 * Grow the decision tree at this node.
	 * Predicts best attributes upon which to split and creates child nodes
	 * 
	 * @param AList<AList<Float>> data
	 * @param Integer targetAttribute
	 * @param ArrayList<Integer> attributeKeys
	 * @return Set<Float> of attribute value decision 
	 */
	public Set<FloatWritable> growDecisionTree(ArrayList<ArrayList<FloatWritable>> data, 
			IntWritable targetAttribute,
			ArrayList<IntWritable> attributeKeys) {
		
		// Initialize value decisions
		Set<FloatWritable> decisions = new HashSet<FloatWritable>();
		
		// If there is no data, return empty decisions list
		if (data.size() == 0) {
			return decisions;
		}
		
		// Values of target attribute
		ArrayList<FloatWritable> targetAttributeValues = new ArrayList<FloatWritable>();
		for (ArrayList<FloatWritable> datapoint : data) {
			targetAttributeValues.add(datapoint.get(targetAttribute.get()));
		}
		
		// Compute majority value
		Float majorityValue = ((FloatWritable)majorityValue(targetAttributeValues)).get();
		
		// If there are no attributes, return empty set
		if (majorityValue == -1.0f) {
			return decisions;
		}
		
		// If only 1 attribute, return default value
		if (attributeKeys.size() == 1) {
			decisions.add(new FloatWritable(majorityValue));
			return decisions;
		}
		
		// If all of the target attributes have one value, then return this value
		// as classification
		if (this.allElementValuesSame(targetAttributeValues)) {
			decisions.add(new FloatWritable(majorityValue));
			return decisions;
		}
		
		// Construction of decision tree node
		else {
		
			// Select best attribute upon which to classify
			IntWritable best = this.chooseSplitAttribute(data, targetAttribute, attributeKeys);
			
			// If computation of best attribute fails, return majority value set
			if (best.get() == -1) {
				decisions.add(new FloatWritable(majorityValue));
				return decisions;
			}
			
			// Set the split attribute of this decision tree
			this.setSplitAttributeKey(best);
			
			// Add each available value of best attribute to decisions
			decisions.addAll(this.getAttributeValues(data, best));
			return decisions;
		}
	}	
	
	/**
	 * majorityValue
	 * 
	 * Compute majority result of target attribute. 
	 * 
	 * @param AList<AList<Float>> 	targetAttributeValues
	 * @return majority				Majority value of target attribute
	 */
	public FloatWritable majorityValue (ArrayList<FloatWritable> targetAttributeValues) {
		
		// Initialize majority value tracker of FloatWritable -> IntegerWritable
		Map<Float, Integer> tracker = new HashMap<Float, Integer>();
		
		// Initialize Majority key
		Float majority = -1.0f;
		
		// Count up occurrences of target value
		for (FloatWritable attributeValue : targetAttributeValues) {
			
			// Add frequency of attribute value to tracker
			if (!tracker.containsKey(attributeValue.get())) {
				
				// Compute frequency of attribute value
				int frequency = Collections.frequency(targetAttributeValues, attributeValue);
				
				// Add frequency to tracker
				tracker.put(attributeValue.get(), frequency);
				
				// Check against majority
				if (!tracker.containsKey(majority) || frequency > tracker.get(majority)) {
					majority = attributeValue.get();
				}
						
			}
	
		}
		
		// Return majority value in this dataset for target attribute
		return new FloatWritable(majority);
	}
	
	/**
	 * allElementValuesSame
	 * 
	 * Tests if all elements in an array are equal
	 * 
	 * @param AList<Float> elements
	 * @return true/false
	 */
	public boolean allElementValuesSame(ArrayList<FloatWritable> elements) {
		
		// Frequency of first element must be equal to size of list
		return Collections.frequency(elements, elements.get(0)) == elements.size();
	
	}
	
	/**
	 * getAttributeValues
	 * 
	 * Get all values of an attribute present in dataset
	 * 
	 * @param <AList<Alist<Float>> data
	 * @param Integer attribute
	 * @return Set<Float> values
	 */
	public Set<FloatWritable> getAttributeValues(ArrayList<ArrayList<FloatWritable>> data, IntWritable attribute) {
		
		// Create set of integers
		Set<FloatWritable> possibleValues = new HashSet<FloatWritable>();
		
		// Add all possible values of target attribute
		for (ArrayList<FloatWritable> datapoint : data) {
			possibleValues.add(datapoint.get(attribute.get()));
		}
		
		// Return set of possible values
		return possibleValues;
	}
	
	/**
	 * chooseSplitAttribute
	 * 
	 * Choose the best attribute upon which to classify
	 * 
	 * @param AList<AList<Float>> data
	 * @param targetAttribute
	 * @param attributes
	 * @return
	 */
	public IntWritable chooseSplitAttribute (ArrayList<ArrayList<FloatWritable>> data,
			IntWritable targetAttribute,
			ArrayList<IntWritable> attributes) {
		
		// Initialize best attribute at -1;
		Integer bestSplitAttribute = -1;
		Float maxInformationGain = 0.0f;
		
		// Compute gain of all attributes except target attribute
		for (IntWritable attribute : attributes) {
			
			// Compute gains and compare with existing maximum 
			if (attribute.get() != targetAttribute.get()) {
				
				// Compute gain from splitting data on this attribute
				Float gain = calculateInformationGain(data, attribute, targetAttribute);
				
				// Compare with existing best information gain, updating best split attribute
				// if greater
				if (gain > maxInformationGain) {
					// Update information gain
					maxInformationGain = gain;
					
					// Update best split attribute
					bestSplitAttribute = attribute.get();
				}
			}
			
		}
		
		return new IntWritable(bestSplitAttribute);
		
	}
	
	/**
	 * calculateEntropy
	 * 
	 * Calculates the entropy associated with a given attribute in the dataset
	 * 
	 * @param AList<AList<Float>> data
	 * @param Integer targetAttribute
	 * @return Double entropy
	 */
	public FloatWritable calculateEntropy(ArrayList<ArrayList<FloatWritable>> subset, IntWritable targetAttribute) {
		
		// Create map of value frequencies for this attribute
		Map<Float, Float> valueFrequency = new HashMap<Float, Float>();
		
		// Initialize entropy at 0
		Float dataEntropy = 0.0f;
		
		// Calculate the frequency of values of the target attribute for each data record
		for (ArrayList<FloatWritable> datapoint : subset) {
			
			// Get value of target attribute at this datapoint
			Float targetValue = ((FloatWritable)datapoint.get(targetAttribute.get())).get();
			
			// If a value for this value exists, increment frequency
			if (valueFrequency.containsKey(targetValue)) {
				valueFrequency.put(targetValue, valueFrequency.get(targetValue) + 1.0f);
				
			// Otherwise, create a new entry with a count of 1
			} else {
				valueFrequency.put(targetValue, 1.0f);
			}
		}
		
		// Calculate the entropy of the data for the target attribute
		for (Float frequency : valueFrequency.values()) {
			dataEntropy += (-frequency/subset.size()) * new Float(Math.log(frequency/subset.size()) / Math.log(2));
		}
		
		return new FloatWritable(dataEntropy);
	}
	
	/**
	 * calculateInformationGain
	 * 
	 * Calculates information gain (decreased entropy) which would result in
	 * splitting the data on the chosen attribute (splitAttribut)
	 *  
	 * @param AList<AList<Float>> data
	 * @param Integer splitAttribute
	 * @param Integer targetAttribute
	 * @return Double infoGain
	 */
	public Float calculateInformationGain(ArrayList<ArrayList<FloatWritable>> data,
			IntWritable splitAttribute, IntWritable targetAttribute) {
		
		// Initialize value frequency
		Map<Float, Float> valueFrequency = new HashMap<Float, Float>();
		
		// Initialize subset entropy
		Float subsetEntropy = 0.0f;
		
		// Calculate frequencies values of split attribute
		for (ArrayList<FloatWritable> datapoint : data) {
			
			// Get target value for split attribute from datapoint
			FloatWritable targetValue = datapoint.get(splitAttribute.get());
			
			// If already existing, increment frequency
			if (valueFrequency.containsKey(targetValue.get())) {
				
				valueFrequency.put(targetValue.get(), valueFrequency.get(targetValue.get()) + 1.0f);
				
			// Otherwise create new entry
			} else {
				valueFrequency.put(targetValue.get(), 1.0f);
			}
		
		}

		// Calculate the sum of the entropies for each of the subsets of datapoints,
		// weighted by their probability of occurring in the training data
		for (Float attributeValue : valueFrequency.keySet()) {
			
			// Calculate probability of this value occurring in the training data
			Float valueProbability = valueFrequency.get(attributeValue) / data.size();
			
			// Create subset of data which only includes records where the split attribute
			// has this attributeValue
			ArrayList<ArrayList<FloatWritable>> subset = 
				getDatapointSubsetByAttributeValue(data, splitAttribute, new FloatWritable(attributeValue));
			
			// Update subset entropy with entropy of this subset relative to the attribute
			// of classification, multiplied by the probability of this value occurring in
			// the training set
			subsetEntropy += valueProbability * calculateEntropy(subset, targetAttribute).get();
			
		}
		
		// Return the difference of the entropy of the whole data set with respect to the 
		// attribute upon which to classify, with the entropy of the split attribute
		return (calculateEntropy(data, targetAttribute).get() - subsetEntropy);
	}
	
	/**
	 * getDatapointSubsetByAttributeValue
	 * 
	 * Returns subset of data for which a particular attribute has a given value
	 * 
	 * @param AList<AList<Float>> data
	 * @param Integer attributeId
	 * @param Float targetValue
	 * @return AList<AList<Float>>
	 */
	public ArrayList<ArrayList<FloatWritable>> getDatapointSubsetByAttributeValue (
			ArrayList<ArrayList<FloatWritable>> data,
			IntWritable attributeId, FloatWritable targetValue) {
		
		// Initialize list of example values
		ArrayList<ArrayList<FloatWritable>> subsetValues = new ArrayList<ArrayList<FloatWritable>>();
		
		// Add only datapoints for which the value of attributeId is attributeValue
		for (ArrayList<FloatWritable> datapoint : data) {
			if (datapoint.get(attributeId.get()).get() == targetValue.get()) {
				subsetValues.add(datapoint);
			}
		}
		
		// Return newly created data list
		return subsetValues;
	}
	
	/**
	 * classifyDatapoint
	 * 
	 * Classify a datapoint with this decision tree
	 * 
	 * @param datapoint
	 * @return targetVertexId
	 */
	public LongWritable classifyDatapoint (ArrayList<Float> datapoint) {
		
		// Attribute on which to base decision
		Integer classifyBy = ((IntWritable) classificationData.get(new Text("classifyBy"))).get();
		
		// Value of split attribute in datapoint
		Float value = datapoint.get(classifyBy);
		
		// Check for attribute value in child nodes
		if (((MapWritable)classificationData.get(new Text("children"))).containsKey(new FloatWritable(value))) {
			
			return (LongWritable) ((MapWritable)classificationData.get(new Text("children"))).get(new FloatWritable(value));
		}
		
		// Return destination as root vertex if needed
		return new LongWritable(-1L);
		
	}
	
	/* End DecisionTreeNode behavior */
	
	/* Begin Data Node Classification Behavior */
	
	/**
	 * verifyClassification
	 * 
	 * Returns whether or not classification matches expected value of target attribute.
	 * 
	 * @param classification
	 */
	public boolean verifyClassification (Float classification) {
		// Get target attribute position
		int targetAttribute = getConf().getInt(TARGET_ATTRIBUTE_POSITION, -1);
	
		// Get expected value of target attribute
		ArrayWritable data = (ArrayWritable) this.classificationData.get(new Text("data"));
		Float expectedValue = ((FloatWritable)data.get()[targetAttribute]).get();
		
		return expectedValue == classification;
	}
	
	/* End Data Node Behavior */
	
	@Override
	public void compute(Iterator<MapWritable> msgIterator)
			throws IOException {
		
		/* Begin Data Vertex Computation */
		
		if (getVertexValue().toString() == "D") {
		
			// First superstep : Training/Testing and Send Training Data
		/*	if (getSuperstep() == 0) {
				
				// First task is to specify own type as vertex with either
				// D or DT (testing) based on position in input set
				int trainingDataSize = getConf().getInt(TRAINING_DATA_CASES, -1);
				
				// Send message to root node with data and vertex type
				if (getVertexId().get() <= trainingDataSize) {
					
					MapWritable trainingData = new MapWritable();
					
					trainingData.put(new Text("vertexType"), new Text("train"));
					trainingData.put(new Text("data"), 
							(ArrayWritable) getVertexValue().get(new Text("data")));
					
					sendMsgToAllEdges(trainingData);
					
					// Training data no longer needed 
					voteToHalt();
				} 
				
				// Set vertex type to testing
				else {
					
					updateVertexValue(new Text("type"), new Text("DT"));
					
				}
			}*/
		}
		
		if (getVertexValue().toString() == "DT") {
		
			// Testing data vertices send message with testing data to root node
			if (getSuperstep() == 1) {
				
				// Initialize message to root node
				MapWritable testingData = new MapWritable();
				
				testingData.put(new Text("vertexType"), new Text("test"));
				testingData.put(new Text("data"), 
						(ArrayWritable) classificationData.get(new Text("data")));
				
				sendMsgToAllEdges(testingData);
				
			} else {
				
				// Waiting for classifications from tree.
				// Once classification received, send result to
				// root treenode, and vote to halt. 
				
				while (msgIterator.hasNext()) {
					
					MapWritable message = msgIterator.next();
					
					if (message.containsKey(new Text("prediction"))) {
						
						MapWritable classificationResult = new MapWritable();
						
						Float prediction = 
							((FloatWritable)message.get(new Text("prediction"))).get();
						
						classificationResult.put(new Text("classified"),
								new BooleanWritable(verifyClassification(prediction)));
						
						sendMsgToAllEdges(classificationResult);
					} 
				
					// Vote to end processing
					voteToHalt();
					
				}
			}
		}
		
		/* End Data Vertex Computation */
		
		/* Begin Tree Vertex Computation */
		
		if (getVertexValue().toString() == "T") {
			
			// On SuperStep 1, root receiving a bunch of training data
			if (getSuperstep() == 1) {
			/*
				ArrayList<ArrayList<Float>> trainingData = new ArrayList<ArrayList<Float>>();
				ArrayList<Integer> attributeKeys = new ArrayList<Integer>();
				
				while (msgIterator.hasNext()) {
					
					MapWritable message = msgIterator.next();
					
					// Add training data from message to trainingData
					if (message.get(new Text("vertexType")).toString() == "train") {
						
						// FIXME
						
					}
				}
				
				// Remove target attribute key from attribute list
				target = ???
						
				for (int i = 0; i < trainingData.get(0).size(); i++) {
					if (i != target)
						attributeKeys.add(i);
				}
				
				// Train root node and add resulting child values
				Set<Float> classifyByValues = growDecisionTree(trainingData, target, attributeKeys);
			
				this.???
				// FIXME addNodesForChildValues
				//this.getNumVertices()
				// */
			}
			
			// On SuperStep 2, root receiving a bunch of testing data, while
			// other vertices might be receiving growing data
			else if (getSuperstep() == 2) {
				
				if (getVertexId().get() == -1L) {
					
					ArrayList<ArrayList<Float>> testingData = new ArrayList<ArrayList<Float>>();
					ArrayList<Integer> attributeKeys = new ArrayList<Integer>();
					
					while (msgIterator.hasNext()) {
						
						MapWritable message = msgIterator.next();
						
						if (message.get(new Text("vertexType")).toString() == "test") {
							
							// FIXME
							
						}
					}
				}
			}
			
			// On later supersteps, wait for data to grow, classify or results
			// messages from testing data vertices
			else {
				
				while (msgIterator.hasNext()) {
					
					MapWritable message = msgIterator.next();
					
					// Grow Decision Tree from this node
					if (message.containsKey(new Text("grow"))) {
						
						// FIXME
						
					} else if (message.containsKey(new Text("result"))) {
						
						// FIXME
						
					} else if (message.containsKey(new Text("classified"))) {
						
						// FIXME
						
					}
				}
			}
			
		}
		
		/* End Tree Vertex Computation */

	}

	@Override
	public int run(String[] args) throws Exception {
		// Initialize command line options menu
		Options options = new Options();
		
		// Add command line options relevant to cluster operation / help
		options.addOption("h", "help", false, "Help");
		options.addOption("w",
				"workers",
				true,
		"Number of workers");
		options.addOption("i",
				"input file",
				true,
		"Input data file");
		options.addOption("o",
				"output",
				true,
		"Output file");
		
		// Add command line options relevant to ID3 decision tree
		options.addOption("n",
				"training cases",
				true,
		"Number of training cases");
		options.addOption("p",
				"target attribute index",
				true,
		"Position of target attribute in input data");
		
		// Command line help formatting and parsing
		HelpFormatter formatter = new HelpFormatter();
		if (args.length == 0) {
			formatter.printHelp(getClass().getName(), options, true);
			return 0;
		}
		
		CommandLineParser parser = new PosixParser();
		CommandLine cmd = parser.parse(options, args);
		if (cmd.hasOption('h')) {
			formatter.printHelp(getClass().getName(), options, true);
			return 0;
		}
		if (!cmd.hasOption('w')) {
			System.out.println("Need to choose the number of workers (-w)");
			return -1;
		}
		if (!cmd.hasOption('i')) {
			System.out.println("Need to set the input training file (-i)");
			return -1;
		}
		if (!cmd.hasOption('n')) {
			System.out.println("Need to set the number of training cases (-n)");
			return -1;
		}
		if (!cmd.hasOption('o')) {
			System.out.println("Need to set the output file (-o)");
			return -1;
		}
		if (!cmd.hasOption('p')) {
			System.out.println("Need to set position of target attribute (-p)");
			return -1;
		}
		
		// Obtain parameters for decision tree construction
		int workers = Integer.parseInt(cmd.getOptionValue('w'));
		
		// Create Giraph Job
		GiraphJob job = new GiraphJob(getConf(), getClass().getName());
		job.setVertexClass(getClass());
		job.setVertexInputFormatClass(DecisionTreeVertexInputFormat.class);
		job.setVertexOutputFormatClass(DecisionTreeVertexOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(cmd.getOptionValue('i')));
		FileOutputFormat.setOutputPath(job, new Path(cmd.getOptionValue('o')));
		job.setWorkerConfiguration(workers, workers, 100.0f);
		
		// Set decision tree training set size
		job.getConfiguration().setInt(TRAINING_DATA_CASES,
				Integer.parseInt(cmd.getOptionValue('n')));
		job.getConfiguration().setInt(TARGET_ATTRIBUTE_POSITION, 
				Integer.parseInt(cmd.getOptionValue('p')));
		
		// Run Giraph Job
		if (job.run(true) == true) {
			return 0;
		} else {
			return -1;
		}
		
	}
	
	public static void main(String[] args) throws Exception {
		System.exit(ToolRunner.run(new DecisionTreeNodeVertex(), args));
	}
	
	/**
	 * InputFormat, OutputFormat, Reader, Writer
	 */
	
	public static class DecisionTreeVertexReader extends
	TextVertexReader<LongWritable, Text, FloatWritable, MapWritable> {
		
		/* Vertices read so far */
		private long verticesRead = 0;
	
		public DecisionTreeVertexReader(
				RecordReader<LongWritable, Text> lineRecordReader) {
			super(lineRecordReader);
		}
	
		@Override
        public boolean nextVertex() throws IOException, InterruptedException {
			return getRecordReader().nextKeyValue();
        }

		public BasicVertex<LongWritable, Text, FloatWritable, MapWritable>
		getCurrentVertex() throws IOException, InterruptedException {
		
			BasicVertex<LongWritable, Text, FloatWritable, MapWritable> vertex = BspUtils
			.<LongWritable, Text, FloatWritable, MapWritable> createVertex(getContext()
					.getConfiguration());
			
			if(verticesRead == 0) {
				
				// Create representative/root vertex
				LongWritable vertexId = new LongWritable(-1L);
				Text vertexValue = new Text("R");
				
				// Storage container for decision tree node
				MapWritable representative = new MapWritable();
				representative.put(new Text("accuracy"), new FloatWritable(0.0f));
				((DecisionTreeNodeVertex)vertex).setData(representative);
				
				// Set edges
				Map<LongWritable, FloatWritable> edges = Maps.newHashMap();
				
				vertex.initialize(vertexId, vertexValue, edges, null);
				verticesRead++;
				return vertex;
			}
		
			Text line = getRecordReader().getCurrentValue();
			try {
				
				// Read data tokens
				StringTokenizer tokenizer = new StringTokenizer(line.toString(), ",");
				LongWritable vertexId = new LongWritable(new Long(verticesRead));
				
				// Initialize data vertex with data
				Text vertexValue = new Text("TR");
				
				MapWritable dataVertex = new MapWritable();
				
				// Read classification datapoints
				FloatWritable[] attributes = new FloatWritable[tokenizer.countTokens()];
				int counter = 0;
				while (tokenizer.hasMoreTokens()) {
					attributes[counter++] = new FloatWritable(Float.parseFloat(tokenizer.nextToken()));
				}
				
				dataVertex.put(new Text("data"), new ArrayWritable(FloatWritable.class, attributes));
				((DecisionTreeNodeVertex)vertex).setData(dataVertex);
				
				// Create edge map
				Map<LongWritable, FloatWritable> edges = Maps.newHashMap();
				edges.put(new LongWritable(-1L), new FloatWritable(1.0f));
				
				vertex.initialize(vertexId, vertexValue, edges, null);
				verticesRead++;
				
			} catch (Exception e) {
				throw new IllegalArgumentException(
						"next: Couldn't get vertex from line " + line, e);
			}
			return vertex;
		
		}
		
		@Override
		public void close() throws IOException {
			super.close();
		}

	}
	
	public static class DecisionTreeVertexInputFormat extends
	TextVertexInputFormat<LongWritable, Text, FloatWritable, MapWritable> {
		
		@Override
		public VertexReader<LongWritable, Text, FloatWritable, MapWritable> 
		createVertexReader(InputSplit split, TaskAttemptContext context) 
		throws IOException {
			
			return new DecisionTreeVertexReader(textInputFormat.createRecordReader(split, context));
		}
	}
	
	public static class DecisionTreeVertexOutputFormat extends
	TextVertexOutputFormat<LongWritable, Text, FloatWritable>{

		@Override
		public VertexWriter<LongWritable, Text, FloatWritable> 
		createVertexWriter(TaskAttemptContext context) throws IOException,
				InterruptedException {
			RecordWriter<Text, Text> recordWriter = textOutputFormat.getRecordWriter(context);
			return new DecisionTreeVertexWriter(recordWriter);
		}

	}
	
	public static class DecisionTreeVertexWriter extends
	TextVertexWriter<LongWritable, Text, FloatWritable> {

		public DecisionTreeVertexWriter(
				RecordWriter<Text, Text> lineRecordWriter) {
			super(lineRecordWriter);
		}

		@Override
		public void writeVertex(
				BasicVertex<LongWritable, Text, FloatWritable, ?> vertex)
		throws IOException, InterruptedException {

			// Only record current vertex if it is a tree node, vs a training/testing set node
			if (vertex.getVertexValue().toString() == "R") {
				
				FloatWritable accuracy = (FloatWritable) ((DecisionTreeNodeVertex)vertex).classificationData.get(new Text("accuracy"));
				
				Text output = new Text("Classified Correctly " + accuracy.get() + "%");
				getRecordWriter().write(new Text(output), null);
				
			}
		}
	}

}
