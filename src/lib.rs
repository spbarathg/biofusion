use serde::{Deserialize, Serialize, ser::SerializeStruct};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug)]
pub struct Colony {
    workers: HashMap<String, Arc<WorkerAnt>>,
    metrics: Arc<Mutex<HashMap<String, serde_json::Value>>>,
}

impl Serialize for Colony {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("Colony", 2)?;
        state.serialize_field("workers", &self.workers)?;
        // For metrics, we'll serialize an empty HashMap since we can't easily serialize the Mutex
        state.serialize_field("metrics", &HashMap::<String, serde_json::Value>::new())?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Colony {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            Workers,
            Metrics,
        }

        struct ColonyVisitor;
        impl<'de> serde::de::Visitor<'de> for ColonyVisitor {
            type Value = Colony;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct Colony")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Colony, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let mut workers = None;
                let mut metrics = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Workers => workers = Some(map.next_value()?),
                        Field::Metrics => metrics = Some(map.next_value()?),
                    }
                }

                let workers = workers.ok_or_else(|| serde::de::Error::missing_field("workers"))?;
                let _metrics = metrics.ok_or_else(|| serde::de::Error::missing_field("metrics"))?;

                Ok(Colony {
                    workers,
                    metrics: Arc::new(Mutex::new(HashMap::new())),
                })
            }
        }

        deserializer.deserialize_struct("Colony", &["workers", "metrics"], ColonyVisitor)
    }
}

// ... rest of the implementation ... 