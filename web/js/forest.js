// Random Forest inference over the exported sklearn model.
//
// The whole point of shipping the real trees rather than reimplementing the
// classifier is that this cannot drift from the desktop app.
// scripts/export_model_json.py verifies the export reproduces sklearn's
// predict_proba exactly before writing the file.

export class Forest {
  constructor(payload) {
    if (payload.schema !== 'gestureflow.forest/1') {
      throw new Error(`Unsupported forest schema: ${payload.schema}`);
    }
    this.classes = payload.classes;
    this.nFeatures = payload.n_features;
    this.leaf = payload.leaf_marker;
    this.trees = payload.trees;
    this.nTrees = payload.trees.length;
  }

  static async load(url) {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Could not load model (${response.status})`);
    }
    return new Forest(await response.json());
  }

  /** Averaged per-tree class probabilities, matching sklearn. */
  predictProba(features) {
    if (features.length !== this.nFeatures) {
      throw new Error(
        `Expected ${this.nFeatures} features, got ${features.length}`,
      );
    }

    const totals = new Array(this.classes.length).fill(0);

    for (let t = 0; t < this.nTrees; t += 1) {
      const tree = this.trees[t];
      const { feature, threshold, left, right, value } = tree;

      let node = 0;
      while (left[node] !== this.leaf) {
        node = features[feature[node]] <= threshold[node]
          ? left[node]
          : right[node];
      }

      const leafValue = value[node];
      for (let c = 0; c < totals.length; c += 1) totals[c] += leafValue[c];
    }

    for (let c = 0; c < totals.length; c += 1) totals[c] /= this.nTrees;
    return totals;
  }

  predict(features) {
    const probs = this.predictProba(features);
    let best = 0;
    for (let i = 1; i < probs.length; i += 1) {
      if (probs[i] > probs[best]) best = i;
    }
    return this.classes[best];
  }
}
