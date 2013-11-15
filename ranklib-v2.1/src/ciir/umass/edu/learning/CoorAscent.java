/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set 
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.utilities.KeyValuePair;
import ciir.umass.edu.utilities.MergeSorter;
import ciir.umass.edu.utilities.SimpleMath;

/**
 * @author vdang
 * 
 * This class implements the linear ranking model known as Coordinate Ascent. It was proposed in this paper:
 *  D. Metzler and W.B. Croft. Linear feature-based models for information retrieval. Information Retrieval, 10(3): 257-274, 2000.
 */
public class CoorAscent extends Ranker {

	//Parameters
	public static int nRestart = 2;
	public static int nMaxIteration = 25;
	public static double stepBase = 0.05;
	public static double stepScale = 2.0;
	public static double tolerance = 0.001;
	public static boolean regularized = false;
	public static double slack = 0.001;//regularized parameter
	
	//Local variables
	protected double[] weight = null; 
	
	protected int current_feature = -1;//used only during learning
	protected double weight_change = -1.0;//used only during learning
	
	public CoorAscent()
	{
		
	}
	public CoorAscent(List<RankList> samples, int[] features)
	{
		super(samples, features);
	}
	
	public void init()
	{
		PRINT("Initializing... ");
		weight = new double[features.length];
		for(int i=0;i<weight.length;i++)
			weight[i] = 1.0f/features.length;
		PRINTLN("[Done]");
	}
	public void learn()
	{
		double[] regVector = new double[weight.length];
		copy(weight, regVector);//uniform weight distribution
		
		//this holds the final best model/score
		double[] bestModel = null;
		double bestModelScore = 0.0;
		
		int[] sign = new int[]{1, -1};
		
		PRINTLN("---------------------------");
		PRINTLN("Training starts...");
		PRINTLN("---------------------------");
		
		for(int r=0;r<nRestart;r++)
		{
			PRINTLN("[+] Random restart #" + (r+1) + "/" + nRestart + "...");
			int consecutive_fails = 0;
			
			//initialize weight vector
			for(int i=0;i<weight.length;i++)
				weight[i] = 1.0f/features.length;
			current_feature = -1;
			double startScore = scorer.score(rank(samples));//compute all the scores (in whatever metric specified) and store them as cache
			
			//local best (within the current restart cycle)
			double bestScore = startScore;
			double[] bestWeight = new double[weight.length];
			copy(weight, bestWeight);

			//There must be at least one feature increasing whose weight helps
			while((weight.length>1&&consecutive_fails < weight.length - 1) || (weight.length==1&&consecutive_fails==0))
			{
				PRINTLN("Shuffling features' order... [Done.]");
				PRINTLN("Optimizing weight vector... ");
				PRINTLN("------------------------------");
				PRINTLN(new int[]{7, 8, 7}, new String[]{"Feature", "weight", scorer.name()});
				PRINTLN("------------------------------");

				int[] fids = getShuffledFeatures();//contain index of elements in the variable @features
				//Try maximizing each feature individually
				for(int i=0;i<fids.length;i++)
				{
					current_feature = fids[i];//this will trigger the "else" branch in the procedure rank()
					
					double origWeight = weight[fids[i]];
					double bestWeightValue = origWeight;//0.0;
					boolean succeeds = false;//whether or not we succeed in finding a better weight value for the current feature
					for(int s=0;s<sign.length;s++)//search by both increasing and decreasing
					{
						double step = 0.001;
						if(origWeight != 0.0 && step > 0.5 * Math.abs(origWeight))
					    	step = stepBase * Math.abs(origWeight);
						double totalStep = step;
						for(int j=0;j<nMaxIteration;j++)
						{
							double w = origWeight + totalStep * sign[s];
							weight_change = w - weight[fids[i]];//weight_change is used in the "else" branch in the procedure rank()
							weight[fids[i]] = w;
							double score = scorer.score(rank(samples));
							if(regularized)
							{
								double penalty = slack * getDistance(weight, regVector);
								score -= penalty;
								//PRINTLN("Penalty: " + penalty);
							}
							if(score > bestScore)//better than the local best, replace the local best with this model
							{
								bestScore = score;
								bestWeightValue = weight[fids[i]];
								succeeds = true;
								
								String bw = ((bestWeightValue>0.0)?"+":"") + SimpleMath.round(bestWeightValue, 4);
								PRINTLN(new int[]{7, 8, 7}, new String[]{features[fids[i]]+"", bw+"", SimpleMath.round(bestScore, 4)+""});
							}
							step *= stepScale;
							totalStep += step;
						}
						if(succeeds)
							break;//no need to search the other direction (e.g. sign = '-')
						else
						{
							weight_change = origWeight - weight[fids[i]];
							updateCached();//restore the cached to reflect the orig. weight for the current feature 
							//so that we can start searching in the other direction (since the optimization in the first direction failed)
							weight[fids[i]] = origWeight;//restore the weight to its initial value
						}
					}
					if(succeeds) 
					{
						weight_change = bestWeightValue - weight[fids[i]];
						updateCached();//restore the cached to reflect the best weight for the current feature
						weight[fids[i]] = bestWeightValue;
						consecutive_fails = 0;//since we found a better weight value
						//then normalize the new weight vector
						double sum = normalize(weight);
						scaleCached(sum);
						copy(weight, bestWeight);
					}
					else
					{
						consecutive_fails++;
						weight_change = origWeight - weight[fids[i]];
						updateCached();//restore the cached to reflect the orig. weight for the current feature since the optimization failed
						//Restore the orig. weight value
						weight[fids[i]] = origWeight;
					}
				}
				PRINTLN("------------------------------");				
				//if we haven't made much progress then quit
				if(bestScore - startScore < tolerance)
					break;
			}
			//update the (global) best model with the best model found in this round
			if(bestModel == null || bestScore > bestModelScore)
			{
				bestModelScore = bestScore;
				bestModel = bestWeight;
			}			
		}
		
		copy(bestModel, weight);
		current_feature = -1;//turn off the cache mode
		scoreOnTrainingData = SimpleMath.round(scorer.score(rank(samples)), 4);
		PRINTLN("---------------------------------");
		PRINTLN("Finished sucessfully.");
		PRINTLN(scorer.name() + " on training data: " + scoreOnTrainingData);
		if(validationSamples != null)
		{
			bestScoreOnValidationData = scorer.score(rank(validationSamples));
			PRINTLN(scorer.name() + " on validation data: " + SimpleMath.round(bestScoreOnValidationData, 4));
		}
		PRINTLN("---------------------------------");
	}
	public RankList rank(RankList rl)
	{
		double[] score = new double[rl.size()];
		if(current_feature == -1)
		{
			for(int i=0;i<rl.size();i++)
			{
				for(int j=0;j<features.length;j++)
					score[i] += weight[j] * rl.get(i).getFeatureValue(features[j]);
				rl.get(i).setCached(score[i]);//use cache of a data point to store its score given the model at this state
			}
		}
		else//This branch is only active during the training process. Here we trade the "clean" codes for efficiency 
		{
			for(int i=0;i<rl.size();i++)
			{
				//cached score = a_1*x_1 + a_2*x_2 + ... + a_n*x_n
				//a_2 ==> a'_2
				//new score = cached score + (a'_2 - a_2)*x_2  ====> NO NEED TO RE-COMPUTE THE WHOLE THING
				score[i] = rl.get(i).getCached() + weight_change * rl.get(i).getFeatureValue(features[current_feature]);
				rl.get(i).setCached(score[i]);
			}
		}
		int[] idx = MergeSorter.sort(score, false); 
		return new RankList(rl, idx);
	}
	public double eval(DataPoint p)
	{
		double score = 0.0;
		for(int i=0;i<features.length;i++)
			score += weight[i] * p.getFeatureValue(features[i]);
		return score;
	}
	public Ranker clone()
	{
		return new CoorAscent();
	}
	public String toString()
	{
		String output = "";
		for(int i=0;i<weight.length;i++)
			output += features[i] + ":" + weight[i] + ((i==weight.length-1)?"":" ");
		return output;
	}
	public String model()
	{
		String output = "## " + name() + "\n";
		output += "## Restart = " + nRestart + "\n";
		output += "## MaxIteration = " + nMaxIteration + "\n";
		output += "## StepBase = " + stepBase + "\n";
		output += "## StepScale = " + stepScale + "\n";
		output += "## Tolerance = " + tolerance + "\n";
		output += "## Regularized = " + regularized + "\n";
		output += "## Slack = " + slack + "\n";
		output += toString();
		return output;
	}
	public void load(String fn)
	{
		try {
			String content = "";
			BufferedReader in = new BufferedReader(
					new InputStreamReader(
							new FileInputStream(fn), "ASCII"));
			
			KeyValuePair kvp = null;
			while((content = in.readLine()) != null)
			{
				content = content.trim();
				if(content.length() == 0)
					continue;
				if(content.indexOf("##")==0)
					continue;
				kvp = new KeyValuePair(content);
				break;
			}
			in.close();
			
			List<String> keys = kvp.keys();
			List<String> values = kvp.values();
			weight = new double[keys.size()];
			features = new int[keys.size()];
			for(int i=0;i<keys.size();i++)
			{
				features[i] = Integer.parseInt(keys.get(i));
				weight[i] = Double.parseDouble(values.get(i));
			}
		}
		catch(Exception ex)
		{
			System.out.println("Error in CoorAscent::load(): " + ex.toString());
		}
	}
	public void printParameters()
	{
		PRINTLN("No. of random restarts: " + nRestart);
		PRINTLN("No. of iterations to search in each direction: " + nMaxIteration);
		PRINTLN("Tolerance: " + tolerance);
		if(regularized)
			PRINTLN("Reg. param: " + slack);
		else
			PRINTLN("Regularization: No");
	}
	public String name()
	{
		return "Coordinate Ascent";
	}
	
	private void updateCached()
	{
		for(int j=0;j<samples.size();j++)
		{
			RankList rl = samples.get(j);
			for(int i=0;i<rl.size();i++)
			{
				//cached score = a_1*x_1 + a_2*x_2 + ... + a_n*x_n
				//a_2 ==> a'_2
				//new score = cached score + (a'_2 - a_2)*x_2  ====> NO NEED TO RE-COMPUTE THE WHOLE THING
				double score = rl.get(i).getCached() + weight_change * rl.get(i).getFeatureValue(features[current_feature]);
				rl.get(i).setCached(score);
			}
		}
	}
	private void scaleCached(double sum)
	{
		for(int j=0;j<samples.size();j++)
		{
			RankList rl = samples.get(j);
			for(int i=0;i<rl.size();i++)
				rl.get(i).setCached(rl.get(i).getCached()/sum);
		}
	}
	private int[] getShuffledFeatures()
	{
		int[] fids = new int[features.length];
		List<Integer> l = new ArrayList<Integer>();
		for(int i=0;i<features.length;i++)
			l.add(i);
		Collections.shuffle(l);
		for(int i=0;i<l.size();i++)
			fids[i] = l.get(i);
		return fids;
	}
	private double getDistance(double[] w1, double[] w2)
	{
		//ASSERT w1.length = w2.length
		double s1 = 0.0;
		double s2 = 0.0;
		for(int i=0;i<w1.length;i++)
		{
			s1 += Math.abs(w1[i]);
			s2 += Math.abs(w2[i]);
		}
		double dist = 0.0;
		for(int i=0;i<w1.length;i++)
		{
			double t = w1[i]/s1 - w2[i]/s2;
			dist += t*t;
		}
		return (double)Math.sqrt(dist);
	}
	private double normalize(double[] weights)
	{
		double sum = 0.0;
		for(int j=0;j<weights.length;j++)
			sum += Math.abs(weights[j]);
		for(int j=0;j<weights.length;j++)
			weights[j] /= sum;
		return sum;
	}

	public void copyModel(CoorAscent ranker)
	{
		weight = new double[features.length];
		if(ranker.weight.length != weight.length)
		{
			System.out.println("These two models use different feature set!!");
			System.exit(1);
		}
		copy(ranker.weight, weight);
		PRINTLN("Model loaded.");
	}
	public double distance(CoorAscent ca)
	{
		return getDistance(weight, ca.weight);
	}
}