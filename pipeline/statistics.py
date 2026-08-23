"""Module for pipeline statistics management and report generation."""

import asyncio
import json
import logging
import time
from pathlib import Path


def check_patent_processed(output_dir: str, patent_id: str) -> bool:
    """
    Check whether a patent was already processed (pipeline_statistics.json exists).
    
    Args:
        output_dir: Results directory
        patent_id: Patent ID
        
    Returns:
        True if the patent was already processed, False otherwise
    """
    patent_dir = Path(output_dir) / patent_id
    stats_file = patent_dir / "pipeline_statistics.json"
    return stats_file.exists()


def load_patent_statistics(output_dir: str, patent_id: str) -> dict:
    """
    Load per-patent statistics from file.
    
    Args:
        output_dir: Results directory
        patent_id: Patent ID
        
    Returns:
        Statistics dictionary, or None if the file was not found
    """
    try:
        patent_dir = Path(output_dir) / patent_id
        stats_file = patent_dir / "pipeline_statistics.json"
        
        if not stats_file.exists():
            return None
            
        with open(stats_file, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    except Exception as e:
        logging.error(f"Error loading statistics for patent {patent_id}: {e}")
        return None


async def save_patent_statistics(
    output_dir: str,
    patent_id: str,
    agent1_usage: dict,
    agent2_usage: dict,
    resolved_count: int,
    unresolved_count: int,
    molecules_with_smiles: int,
    verified_molecules: int,
    unverified_molecules: int,
    processing_time: float,
    timestamp: float
):
    """
    Save statistics for a single patent to a JSON file.
    Enables progress tracking and recovery after interruption.
    
    Args:
        output_dir: Directory for saving results
        patent_id: Patent ID
        agent1_usage: Agent 1 usage statistics
        agent2_usage: Agent 2 usage statistics
        resolved_count: Number of resolved aliases
        unresolved_count: Number of unresolved aliases
        molecules_with_smiles: Number of molecules with SMILES
        verified_molecules: Number of verified molecules
        unverified_molecules: Number of unverified molecules
        processing_time: Patent processing time (seconds)
        timestamp: Processing completion timestamp
    """
    try:
        patent_dir = Path(output_dir) / patent_id
        patent_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare per-patent statistics
        patent_stats = {
            "patent_id": patent_id,
            "processing_info": {
                "timestamp": timestamp,
                "processing_time_seconds": processing_time,
                "processing_time_formatted": f"{int(processing_time // 60):02d}:{processing_time % 60:06.3f}"
            },
            "agent1_stats": {
                "prompt_tokens": agent1_usage.get("prompt_tokens", 0),
                "completion_tokens": agent1_usage.get("completion_tokens", 0),
                "total_tokens": agent1_usage.get("total_tokens", 0),
                "cost": agent1_usage.get("cost", 0),
                "request_count": agent1_usage.get("request_count", 0)
            },
            "agent2_stats": {
                "prompt_tokens": agent2_usage.get("prompt_tokens", 0),
                "completion_tokens": agent2_usage.get("completion_tokens", 0),
                "total_tokens": agent2_usage.get("total_tokens", 0),
                "cost": agent2_usage.get("cost", 0),
                "request_count": agent2_usage.get("request_count", 0),
                "pyopsin_filtered_count": agent2_usage.get("pyopsin_filtered_count", 0)
            },
            "data_counts": {
                "aliases_resolved": resolved_count,
                "aliases_unresolved": unresolved_count,
                "molecules_with_smiles": molecules_with_smiles,
                "molecules_verified": verified_molecules,
                "molecules_unverified": unverified_molecules
            },
            "total_stats": {
                "prompt_tokens": agent1_usage.get("prompt_tokens", 0) + agent2_usage.get("prompt_tokens", 0),
                "completion_tokens": agent1_usage.get("completion_tokens", 0) + agent2_usage.get("completion_tokens", 0),
                "total_tokens": agent1_usage.get("total_tokens", 0) + agent2_usage.get("total_tokens", 0),
                "cost": agent1_usage.get("cost", 0) + agent2_usage.get("cost", 0),
                "request_count": agent1_usage.get("request_count", 0) + agent2_usage.get("request_count", 0)
            }
        }
        
        # Save to a JSON file
        stats_file = patent_dir / "pipeline_statistics.json"
        await asyncio.to_thread(
            lambda: stats_file.write_text(
                json.dumps(patent_stats, indent=2, ensure_ascii=False),
                encoding='utf-8'
            )
        )
        
        logging.debug(f"Patent statistics for {patent_id} saved to {stats_file}")
        
    except Exception as e:
        logging.error(f"Error saving statistics for patent {patent_id}: {e}")


async def aggregate_patent_statistics(output_dir: str) -> dict:
    """
    Aggregate statistics from all per-patent pipeline_statistics.json files.
    Useful for recovering global statistics after interruption.
    
    Args:
        output_dir: Results directory
        
    Returns:
        Aggregated statistics in global_stats format
    """
    aggregated = {
        "agent1": {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0,
            "request_count": 0, "token_per_request": [], "max_tokens_per_request": 0,
            "completion_tokens_per_request": [], "max_completion_tokens_per_request": 0
        },
        "agent2": {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0,
            "request_count": 0, "token_per_request": [], "max_tokens_per_request": 0,
            "completion_tokens_per_request": [], "max_completion_tokens_per_request": 0,
            "pyopsin_filtered_count": 0
        },
        "total": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
        "data_counts": {
            "patents_processed": 0,
            "bioactivity_data_found": 0,
            "aliases_resolved": 0,
            "aliases_unresolved": 0,
            "molecules_resolved_with_smiles": 0,
            "molecules_resolved_verified": 0,
            "molecules_resolved_unverified": 0
        }
    }
    
    try:
        output_path = Path(output_dir)
        patent_dirs = [d for d in output_path.iterdir() if d.is_dir()]
        
        patents_found = 0
        for patent_dir in patent_dirs:
            stats_file = patent_dir / "pipeline_statistics.json"
            if not stats_file.exists():
                continue
                
            try:
                patent_stats = await asyncio.to_thread(
                    lambda: json.loads(stats_file.read_text(encoding='utf-8'))
                )
                
                patents_found += 1
                
                # Aggregate Agent 1
                agent1 = patent_stats.get("agent1_stats", {})
                for key in ["prompt_tokens", "completion_tokens", "total_tokens", "cost", "request_count"]:
                    aggregated["agent1"][key] += agent1.get(key, 0)
                
                # Aggregate Agent 2
                agent2 = patent_stats.get("agent2_stats", {})
                for key in ["prompt_tokens", "completion_tokens", "total_tokens", "cost", "request_count"]:
                    aggregated["agent2"][key] += agent2.get(key, 0)
                aggregated["agent2"]["pyopsin_filtered_count"] += agent2.get("pyopsin_filtered_count", 0)
                
                # Aggregate data counts
                data = patent_stats.get("data_counts", {})
                aggregated["data_counts"]["patents_processed"] += 1
                aggregated["data_counts"]["aliases_resolved"] += data.get("aliases_resolved", 0)
                aggregated["data_counts"]["aliases_unresolved"] += data.get("aliases_unresolved", 0)
                aggregated["data_counts"]["molecules_resolved_with_smiles"] += data.get("molecules_with_smiles", 0)
                aggregated["data_counts"]["molecules_resolved_verified"] += data.get("molecules_verified", 0)
                aggregated["data_counts"]["molecules_resolved_unverified"] += data.get("molecules_unverified", 0)
                
            except Exception as e:
                logging.warning(f"Failed to load statistics from {stats_file}: {e}")
                continue
        
        # Compute totals
        for key in ["prompt_tokens", "completion_tokens", "total_tokens", "cost"]:
            aggregated["total"][key] = aggregated["agent1"][key] + aggregated["agent2"][key]
        
        logging.info(f"Aggregated statistics from {patents_found} patents")
        return aggregated
        
    except Exception as e:
        logging.error(f"Error aggregating patent statistics: {e}")
        return aggregated


async def save_statistics_to_files(output_dir: str, global_stats: dict, total_time: float, start_time: float, debug_mode: bool = False):
    """
    Save statistics to JSON and text files for further analysis.
    """
    try:
        # Prepare data for saving
        statistics = {
            "execution_info": {
                "start_time": start_time,
                "end_time": time.time(),
                "total_time_seconds": total_time,
                "total_time_formatted": f"{int(total_time // 3600):02d}:{int((total_time % 3600) // 60):02d}:{total_time % 60:06.3f}",
                "debug_mode_enabled": debug_mode
            },
            "data_counts": global_stats["data_counts"].copy(),
            "agent1_stats": global_stats["agent1"].copy(),
            "agent2_stats": global_stats["agent2"].copy(),
            "total_stats": global_stats["total"].copy(),
            "error_analysis": global_stats.get("error_analysis", {}),
            "efficiency_metrics": {}
        }
        
        # Compute efficiency metrics
        total_requests = global_stats['agent1']['request_count'] + global_stats['agent2']['request_count']
        resolved_aliases = global_stats['data_counts']['aliases_resolved']
        patents_processed = global_stats['data_counts']['patents_processed']
        
        statistics["total_stats"]["request_count"] = total_requests
        
        # Add extended efficiency metrics
        if patents_processed > 0:
            statistics["efficiency_metrics"]["patents_per_hour"] = patents_processed / (total_time / 3600) if total_time > 0 else 0
            statistics["efficiency_metrics"]["bioactivity_data_per_patent"] = global_stats['data_counts']['bioactivity_data_found'] / patents_processed
            statistics["efficiency_metrics"]["cost_per_patent"] = global_stats['total']['cost'] / patents_processed
            statistics["efficiency_metrics"]["tokens_per_patent"] = global_stats['total']['total_tokens'] / patents_processed
        
        if total_requests > 0:
            statistics["total_stats"]["avg_cost_per_request"] = global_stats['total']['cost'] / total_requests
            statistics["total_stats"]["avg_time_per_request"] = total_time / total_requests
        
        if resolved_aliases > 0:
            statistics["efficiency_metrics"]["tokens_per_resolved_alias"] = global_stats['total']['total_tokens'] / resolved_aliases
            statistics["efficiency_metrics"]["time_per_resolved_alias"] = total_time / resolved_aliases
            statistics["efficiency_metrics"]["cost_per_resolved_alias"] = global_stats['total']['cost'] / resolved_aliases
            statistics["efficiency_metrics"]["theoretical_connections_per_hour"] = 3600 / (total_time / resolved_aliases) if total_time > 0 else 0
        
        # Add success-rate metrics
        total_aliases = global_stats['data_counts']['aliases_resolved'] + global_stats['data_counts']['aliases_unresolved']
        if total_aliases > 0:
            statistics["efficiency_metrics"]["alias_resolution_rate_percent"] = (global_stats['data_counts']['aliases_resolved'] / total_aliases) * 100
        
        # Compute total error count and error rate
        if "error_analysis" in statistics and statistics["error_analysis"]:
            agent1_errors = statistics["error_analysis"]["agent1_errors"]
            agent2_errors = statistics["error_analysis"]["agent2_errors"]
            
            total_agent1_errors = sum(agent1_errors.values())
            total_agent2_errors = sum(agent2_errors.values())
            
            statistics["error_analysis"]["total_agent1_errors"] = total_agent1_errors
            statistics["error_analysis"]["total_agent2_errors"] = total_agent2_errors
            statistics["error_analysis"]["total_errors"] = total_agent1_errors + total_agent2_errors
            
            if total_requests > 0:
                statistics["error_analysis"]["error_rate_percent"] = (statistics["error_analysis"]["total_errors"] / total_requests) * 100
        
        # Drop agent2 token lists from JSON to save space (they are very large)
        # Keep agent1 lists: only 3 stages and useful for detailed statistics
        if "token_per_request" in statistics["agent2_stats"]:
            del statistics["agent2_stats"]["token_per_request"]
        if "completion_tokens_per_request" in statistics["agent2_stats"]:
            del statistics["agent2_stats"]["completion_tokens_per_request"]
        
        # Save JSON file
        stats_json_file = Path(output_dir) / "pipeline_statistics.json"
        await asyncio.to_thread(
            lambda: stats_json_file.write_text(
                json.dumps(statistics, indent=2, ensure_ascii=False),
                encoding='utf-8'
            )
        )
        
        # Save text report
        stats_txt_file = Path(output_dir) / "pipeline_statistics.txt"
        report_content = generate_text_report(statistics)
        await asyncio.to_thread(
            lambda: stats_txt_file.write_text(report_content, encoding='utf-8')
        )
        
        logging.info(f"Statistics saved to files:")
        logging.info(f"  - JSON: {stats_json_file}")
        logging.info(f"  - Text: {stats_txt_file}")
        
    except Exception as e:
        logging.error(f"Error saving statistics: {e}")


def generate_text_report(statistics: dict) -> str:
    """
    Generate a text report from statistics.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("FINAL TOKEN AND DATA USAGE STATISTICS")
    lines.append("=" * 70)
    
    # Execution time
    lines.append(f"EXECUTION TIME:")
    lines.append(f"  - Total time: {statistics['execution_info']['total_time_formatted']}")
    lines.append(f"  - Start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(statistics['execution_info']['start_time']))}")
    lines.append(f"  - End: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(statistics['execution_info']['end_time']))}")
    lines.append("")
    
    # Found data
    data_counts = statistics["data_counts"]
    lines.append(f"FOUND DATA:")
    lines.append(f"  - Patents processed: {data_counts['patents_processed']:,}")
    lines.append(f"  - Bioactivity data points found: {data_counts['bioactivity_data_found']:,}")
    lines.append(f"  - Aliases resolved: {data_counts['aliases_resolved']:,}")
    lines.append(f"  - Aliases unresolved: {data_counts['aliases_unresolved']:,}")
    
    total_aliases = data_counts['aliases_resolved'] + data_counts['aliases_unresolved']
    if total_aliases > 0:
        resolve_rate = data_counts['aliases_resolved'] / total_aliases * 100
        lines.append(f"  - Alias resolution rate: {resolve_rate:.1f}%")
    
    # NEW STATISTICS: SMILES and verification
    lines.append(f"  - Molecules with SMILES strings: {data_counts['molecules_resolved_with_smiles']:,}")
    if data_counts['aliases_resolved'] > 0:
        smiles_rate = data_counts['molecules_resolved_with_smiles'] / data_counts['aliases_resolved'] * 100
        lines.append(f"  - Structure resolution rate: {smiles_rate:.1f}%")
    lines.append(f"  - Verified resolutions: {data_counts['molecules_resolved_verified']:,}")
    lines.append(f"  - Unverified resolutions: {data_counts['molecules_resolved_unverified']:,}")
    total_resolved_methods = data_counts['molecules_resolved_verified'] + data_counts['molecules_resolved_unverified']
    if total_resolved_methods > 0:
        verified_rate = data_counts['molecules_resolved_verified'] / total_resolved_methods * 100
        lines.append(f"  - Verified resolution rate: {verified_rate:.1f}%")
    lines.append("")
    
    # Agent 1 statistics
    agent1 = statistics["agent1_stats"]
    lines.append(f"AGENT 1 (Data extraction):")
    lines.append(f"  - LLM requests: {agent1['request_count']:,}")
    lines.append(f"  - Prompt tokens: {agent1['prompt_tokens']:,}")
    lines.append(f"  - Completion tokens: {agent1['completion_tokens']:,}")
    lines.append(f"  - Total tokens: {agent1['total_tokens']:,}")
    if agent1['request_count'] > 0:
        avg_tokens = agent1['total_tokens'] / agent1['request_count']
        avg_completion = agent1['completion_tokens'] / agent1['request_count']
        lines.append(f"  - Average tokens per request: {avg_tokens:.1f}")
        lines.append(f"  - Average completion tokens per request: {avg_completion:.1f}")
    lines.append(f"  - Maximum tokens per request: {agent1['max_tokens_per_request']:,}")
    lines.append(f"  - Maximum completion tokens per request: {agent1['max_completion_tokens_per_request']:,}")
    lines.append(f"  - Cost: ${agent1['cost']:.4f}")
    
    # Detailed per-stage statistics (if available)
    if 'token_per_request' in agent1 and len(agent1['token_per_request']) >= 3:
        lines.append("")
        token_list = agent1['token_per_request']
        completion_list = agent1['completion_tokens_per_request']
        total_cost = agent1['cost']
        total_tokens = agent1['total_tokens']
        
        # Determine patent count (request count / 3)
        num_patents = len(token_list) // 3
        
        if num_patents == 1:
            # Single patent: show detailed per-stage statistics
            lines.append("  Detailed per-stage statistics:")
            for i in range(3):
                stage_name = f"Stage {i+1}"
                stage_tokens = token_list[i]
                stage_completion = completion_list[i] if i < len(completion_list) else 0
                stage_prompt = stage_tokens - stage_completion
                stage_cost = (stage_tokens / total_tokens * total_cost) if total_tokens > 0 else 0
                
                lines.append(f"    {stage_name}:")
                lines.append(f"      • Prompt tokens: {stage_prompt:,}")
                lines.append(f"      • Completion tokens: {stage_completion:,}")
                lines.append(f"      • Total tokens: {stage_tokens:,}")
                lines.append(f"      • Cost: ${stage_cost:.4f}")
        else:
            # Multiple patents: show average and maximum values per stage
            lines.append(f"  Per-stage statistics ({num_patents} patents):")
            
            for stage_idx in range(3):
                # Collect data for this stage across all patents
                stage_tokens_list = [token_list[i*3 + stage_idx] for i in range(num_patents) if i*3 + stage_idx < len(token_list)]
                stage_completion_list = [completion_list[i*3 + stage_idx] for i in range(num_patents) if i*3 + stage_idx < len(completion_list)]
                
                if stage_tokens_list:
                    # Average values
                    avg_total = sum(stage_tokens_list) / len(stage_tokens_list)
                    avg_completion = sum(stage_completion_list) / len(stage_completion_list) if stage_completion_list else 0
                    avg_prompt = avg_total - avg_completion
                    
                    # Maximum values
                    max_total = max(stage_tokens_list)
                    max_idx = stage_tokens_list.index(max_total)
                    max_completion = stage_completion_list[max_idx] if max_idx < len(stage_completion_list) else 0
                    max_prompt = max_total - max_completion
                    
                    # Stage cost = sum of all requests for this stage
                    stage_total_tokens = sum(stage_tokens_list)
                    stage_cost = (stage_total_tokens / total_tokens * total_cost) if total_tokens > 0 else 0
                    
                    stage_name = f"Stage {stage_idx+1}"
                    lines.append(f"    {stage_name}:")
                    lines.append(f"      • Average: {avg_total:,.0f} tokens (prompt: {avg_prompt:,.0f}, completion: {avg_completion:,.0f})")
                    lines.append(f"      • Maximum: {max_total:,} tokens (prompt: {max_prompt:,}, completion: {max_completion:,})")
                    lines.append(f"      • Cost (sum): ${stage_cost:.4f}")
    
    lines.append("")
    
    # Agent 2 statistics
    agent2 = statistics["agent2_stats"]
    lines.append(f"AGENT 2 (Alias resolution):")
    lines.append(f"  - LLM requests: {agent2['request_count']:,}")
    lines.append(f"  - Prompt tokens: {agent2['prompt_tokens']:,}")
    lines.append(f"  - Completion tokens: {agent2['completion_tokens']:,}")
    lines.append(f"  - Total tokens: {agent2['total_tokens']:,}")
    if agent2['request_count'] > 0:
        avg_tokens = agent2['total_tokens'] / agent2['request_count']
        avg_completion = agent2['completion_tokens'] / agent2['request_count']
        lines.append(f"  - Average tokens per request: {avg_tokens:.1f}")
        lines.append(f"  - Average completion tokens per request: {avg_completion:.1f}")
    lines.append(f"  - Maximum tokens per request: {agent2['max_tokens_per_request']:,}")
    lines.append(f"  - Maximum completion tokens per request: {agent2['max_completion_tokens_per_request']:,}")
    lines.append(f"  - Cost: ${agent2['cost']:.4f}")
    lines.append(f"  - PyOpsin filtered: {agent2['pyopsin_filtered_count']:,}")
    lines.append("")
    
    # Overall total
    total_stats = statistics["total_stats"]
    lines.append(f"OVERALL TOTAL:")
    lines.append(f"  - Total LLM requests: {total_stats['request_count']:,}")
    lines.append(f"  - Total tokens: {total_stats['total_tokens']:,}")
    lines.append(f"  - Total cost: ${total_stats['cost']:.4f}")
    if 'avg_cost_per_request' in total_stats:
        lines.append(f"  - Average cost per request: ${total_stats['avg_cost_per_request']:.6f}")
    if 'avg_time_per_request' in total_stats:
        lines.append(f"  - Average time per request: {total_stats['avg_time_per_request']:.3f} sec")
    lines.append("")
    
    # Add error section
    if "error_analysis" in statistics and statistics["error_analysis"]:
        error_analysis = statistics["error_analysis"]
        lines.append(f"ERROR ANALYSIS:")
        
        if "agent1_errors" in error_analysis:
            agent1_errors = error_analysis["agent1_errors"]
            total_agent1_errors = error_analysis.get("total_agent1_errors", 0)
            if total_agent1_errors > 0:
                lines.append(f"  Agent 1 errors (total: {total_agent1_errors}):")
                for error_type, count in agent1_errors.items():
                    if count > 0:
                        lines.append(f"    - {error_type}: {count}")
        
        if "agent2_errors" in error_analysis:
            agent2_errors = error_analysis["agent2_errors"]
            total_agent2_errors = error_analysis.get("total_agent2_errors", 0)
            if total_agent2_errors > 0:
                lines.append(f"  Agent 2 errors (total: {total_agent2_errors}):")
                for error_type, count in agent2_errors.items():
                    if count > 0:
                        lines.append(f"    - {error_type}: {count}")
        
        if "error_rate_percent" in error_analysis:
            lines.append(f"  Overall error rate: {error_analysis['error_rate_percent']:.2f}%")
        
        total_errors = error_analysis.get("total_errors", 0)
        if total_errors == 0:
            lines.append(f"  No errors detected - great work!")
        lines.append("")
    
    # Key efficiency metrics
    if statistics["efficiency_metrics"]:
        eff = statistics["efficiency_metrics"]
        lines.append(f"KEY EFFICIENCY METRICS:")
        
        # Add new metrics
        if 'patents_per_hour' in eff:
            lines.append(f"  - Patents per hour: {eff['patents_per_hour']:.1f}")
        if 'bioactivity_data_per_patent' in eff:
            lines.append(f"  - Bioactivity data points per patent: {eff['bioactivity_data_per_patent']:.1f}")
        if 'cost_per_patent' in eff:
            lines.append(f"  - Cost per patent: ${eff['cost_per_patent']:.4f}")
        if 'tokens_per_patent' in eff:
            lines.append(f"  - Tokens per patent: {eff['tokens_per_patent']:.0f}")
        
        # Existing metrics
        if 'tokens_per_resolved_alias' in eff:
            lines.append(f"  - Tokens per resolved connection: {eff['tokens_per_resolved_alias']:.1f}")
        if 'time_per_resolved_alias' in eff:
            lines.append(f"  - Time per connection: {eff['time_per_resolved_alias']:.3f} sec")
        if 'cost_per_resolved_alias' in eff:
            lines.append(f"  - Cost per connection: ${eff['cost_per_resolved_alias']:.6f}")
        if 'theoretical_connections_per_hour' in eff:
            lines.append(f"  - Connections per hour (theoretical): {eff['theoretical_connections_per_hour']:.0f}")
        
        # Percentage metrics
        if 'alias_resolution_rate_percent' in eff:
            lines.append(f"  - Alias resolution rate: {eff['alias_resolution_rate_percent']:.1f}%")
    else:
        lines.append(f"KEY EFFICIENCY METRICS:")
        lines.append(f"  - No connections found - cannot compute efficiency metrics")
    
    lines.append("=" * 70)
    return "\n".join(lines)



