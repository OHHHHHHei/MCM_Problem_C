"""
数据处理模块
用于加载和预处理 DWTS 数据集
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class Contestant:
    """选手数据结构"""
    name: str
    partner: str
    industry: str
    homestate: Optional[str]
    homecountry: str
    age: int
    season: int
    result: str
    placement: int
    weekly_scores: Dict[int, List[float]]  # week -> [judge1, judge2, judge3, judge4]


@dataclass
class EliminationEvent:
    """淘汰事件"""
    season: int
    week: int
    eliminated: str  # 被淘汰选手名称
    survivors: List[str]  # 幸存者名称列表
    bottom_two: Optional[Tuple[str, str]] = None  # 对于S28+的 Judge Save


class DataProcessor:
    """数据预处理类"""
    
    def __init__(self, csv_path: str):
        """初始化数据处理器"""
        self.df = pd.read_csv(csv_path)
        self.contestants = {}
        self.seasons_data = {}
        self._process_data()
    
    def _process_data(self):
        """处理原始数据"""
        for idx, row in self.df.iterrows():
            try:
                # 解析基本信息
                name = str(row['celebrity_name']).strip()
                season = int(row['season'])
                
                # 解析每周分数
                weekly_scores = {}
                for week in range(1, 12):  # 最多11周
                    scores = []
                    for judge in range(1, 5):  # 4个评委
                        col = f'week{week}_judge{judge}_score'
                        if col in row.index:
                            val = row[col]
                            if pd.notna(val) and val != 'N/A' and val != 0:
                                try:
                                    scores.append(float(val))
                                except:
                                    pass
                    if scores and any(s > 0 for s in scores):
                        weekly_scores[week] = scores
                
                # 解析排名
                placement = row['placement']
                if pd.isna(placement):
                    placement = 99
                else:
                    try:
                        placement = int(placement)
                    except:
                        placement = 99
                
                # 解析年龄
                age = row['celebrity_age_during_season']
                if pd.isna(age):
                    age = 30  # 默认年龄
                else:
                    try:
                        age = int(float(age))
                    except:
                        age = 30
                
                contestant = Contestant(
                    name=name,
                    partner=str(row['ballroom_partner']) if pd.notna(row['ballroom_partner']) else '',
                    industry=str(row['celebrity_industry']) if pd.notna(row['celebrity_industry']) else '',
                    homestate=str(row['celebrity_homestate']) if pd.notna(row['celebrity_homestate']) else None,
                    homecountry=str(row['celebrity_homecountry/region']) if pd.notna(row['celebrity_homecountry/region']) else 'Unknown',
                    age=age,
                    season=season,
                    result=str(row['results']) if pd.notna(row['results']) else '',
                    placement=placement,
                    weekly_scores=weekly_scores
                )
                
                key = (name, season)
                self.contestants[key] = contestant
                
                # 按赛季组织
                if season not in self.seasons_data:
                    self.seasons_data[season] = []
                self.seasons_data[season].append(contestant)
                
            except Exception as e:
                print(f"Warning: Error processing row {idx}: {e}")
                continue
    
    def get_seasons(self) -> List[int]:
        """获取所有赛季列表"""
        return sorted(self.seasons_data.keys())
    
    def get_contestants_in_season(self, season: int) -> List[Contestant]:
        """获取某赛季所有选手"""
        return self.seasons_data.get(season, [])
    
    def get_active_contestants(self, season: int, week: int) -> List[Contestant]:
        """获取某赛季某周仍在场的选手"""
        contestants = self.get_contestants_in_season(season)
        active = []
        for c in contestants:
            if week in c.weekly_scores:
                # 检查是否有有效分数
                scores = c.weekly_scores[week]
                if scores and any(s > 0 for s in scores):
                    active.append(c)
        return active
    
    def get_elimination_week(self, contestant: Contestant) -> Optional[int]:
        """获取选手被淘汰的周数"""
        result = contestant.result.lower()
        
        if 'withdrew' in result:
            return None  # 主动退出不计入
        
        match = re.search(r'week\s*(\d+)', result, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # 如果是冠军/亚军等，返回None表示未被淘汰
        if any(x in result.lower() for x in ['1st', '2nd', '3rd', '4th', '5th', 'place']):
            return None
        
        return None
    
    def get_weekly_total_score(self, contestant: Contestant, week: int) -> float:
        """获取选手某周的总分"""
        if week not in contestant.weekly_scores:
            return 0.0
        scores = contestant.weekly_scores[week]
        return sum(scores) if scores else 0.0
    
    def get_weekly_average_score(self, contestant: Contestant, week: int) -> float:
        """获取选手某周的平均分"""
        if week not in contestant.weekly_scores:
            return 0.0
        scores = contestant.weekly_scores[week]
        return np.mean(scores) if scores else 0.0
    
    def get_elimination_events(self, season: int) -> List[EliminationEvent]:
        """获取某赛季所有淘汰事件"""
        contestants = self.get_contestants_in_season(season)
        events = []
        
        # 找出每周被淘汰的选手
        eliminations_by_week = {}
        for c in contestants:
            elim_week = self.get_elimination_week(c)
            if elim_week is not None:
                if elim_week not in eliminations_by_week:
                    eliminations_by_week[elim_week] = []
                eliminations_by_week[elim_week].append(c.name)
        
        # 构建淘汰事件
        for week in sorted(eliminations_by_week.keys()):
            eliminated_names = eliminations_by_week[week]
            
            # 获取该周仍在场的所有选手
            active = self.get_active_contestants(season, week)
            active_names = [c.name for c in active]
            
            for elim_name in eliminated_names:
                survivors = [n for n in active_names if n != elim_name]
                events.append(EliminationEvent(
                    season=season,
                    week=week,
                    eliminated=elim_name,
                    survivors=survivors
                ))
        
        return events
    
    def get_rule_type(self, season: int) -> str:
        """判断赛季使用的规则类型"""
        if season <= 2:
            return 'RANK'
        elif season <= 27:
            return 'PERCENTAGE'
        else:
            return 'RANK_WITH_SAVE'  # S28+ 有 Judge Save
    
    def get_feature_vector(self, contestant: Contestant) -> np.ndarray:
        """生成选手特征向量 (用于初始化回归)"""
        # 行业编码
        industries = ['Actor/Actress', 'Athlete', 'Singer/Rapper', 'TV Personality', 'Model', 'Comedian', 'Other']
        industry_code = np.zeros(len(industries))
        for i, ind in enumerate(industries):
            if ind.lower() in contestant.industry.lower():
                industry_code[i] = 1
                break
        if industry_code.sum() == 0:
            industry_code[-1] = 1  # Other
        
        # 标准化年龄
        age_norm = (contestant.age - 35) / 15  # 均值约35，标准差约15
        
        # 是否来自美国
        is_us = 1.0 if contestant.homecountry.lower() == 'united states' else 0.0
        
        features = np.concatenate([
            [age_norm, is_us],
            industry_code
        ])
        
        return features
    
    def compute_zscore_scores(self, season: int, week: int) -> Dict[str, float]:
        """计算该周所有选手的标准化评审分 (Z-score)"""
        active = self.get_active_contestants(season, week)
        
        scores = {}
        for c in active:
            scores[c.name] = self.get_weekly_average_score(c, week)
        
        if not scores:
            return {}
        
        score_values = list(scores.values())
        mean_score = np.mean(score_values)
        std_score = np.std(score_values) if len(score_values) > 1 else 1.0
        
        if std_score < 0.01:
            std_score = 1.0
        
        return {name: (s - mean_score) / std_score for name, s in scores.items()}
    
    def export_cleaned_data(self, output_dir: str = 'cleaned_data'):
        """
        导出清洗后的数据到CSV文件
        
        生成文件:
            - cleaned_contestants.csv: 选手基本信息
            - weekly_scores.csv: 每周评审分
            - elimination_events.csv: 淘汰事件列表
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 导出选手基本信息
        contestants_rows = []
        for (name, season), c in self.contestants.items():
            contestants_rows.append({
                'season': c.season,
                'name': c.name,
                'partner': c.partner,
                'industry': c.industry,
                'homestate': c.homestate or '',
                'homecountry': c.homecountry,
                'age': c.age,
                'result': c.result,
                'placement': c.placement,
                'rule_type': self.get_rule_type(c.season),
                'elimination_week': self.get_elimination_week(c) or '',
                'total_weeks': len(c.weekly_scores)
            })
        
        df_contestants = pd.DataFrame(contestants_rows)
        df_contestants = df_contestants.sort_values(['season', 'placement'])
        df_contestants.to_csv(f'{output_dir}/cleaned_contestants.csv', index=False, encoding='utf-8-sig')
        print(f"导出选手信息: {len(contestants_rows)} 条 -> {output_dir}/cleaned_contestants.csv")
        
        # 2. 导出每周评审分
        scores_rows = []
        for (name, season), c in self.contestants.items():
            for week, scores in c.weekly_scores.items():
                scores_rows.append({
                    'season': c.season,
                    'week': week,
                    'name': c.name,
                    'judge1_score': scores[0] if len(scores) > 0 else '',
                    'judge2_score': scores[1] if len(scores) > 1 else '',
                    'judge3_score': scores[2] if len(scores) > 2 else '',
                    'judge4_score': scores[3] if len(scores) > 3 else '',
                    'total_score': sum(scores),
                    'average_score': np.mean(scores),
                    'num_judges': len(scores)
                })
        
        df_scores = pd.DataFrame(scores_rows)
        df_scores = df_scores.sort_values(['season', 'week', 'total_score'], ascending=[True, True, False])
        df_scores.to_csv(f'{output_dir}/weekly_scores.csv', index=False, encoding='utf-8-sig')
        print(f"导出评审分: {len(scores_rows)} 条 -> {output_dir}/weekly_scores.csv")
        
        # 3. 导出淘汰事件
        events_rows = []
        for season in self.get_seasons():
            events = self.get_elimination_events(season)
            for e in events:
                events_rows.append({
                    'season': e.season,
                    'week': e.week,
                    'eliminated': e.eliminated,
                    'num_survivors': len(e.survivors),
                    'survivors': '; '.join(e.survivors),
                    'rule_type': self.get_rule_type(e.season)
                })
        
        df_events = pd.DataFrame(events_rows)
        df_events = df_events.sort_values(['season', 'week'])
        df_events.to_csv(f'{output_dir}/elimination_events.csv', index=False, encoding='utf-8-sig')
        print(f"导出淘汰事件: {len(events_rows)} 条 -> {output_dir}/elimination_events.csv")
        
        # 4. 导出Z-Score标准化分数
        zscore_rows = []
        for season in self.get_seasons():
            contestants = self.get_contestants_in_season(season)
            max_week = max(max(c.weekly_scores.keys()) for c in contestants if c.weekly_scores)
            
            for week in range(1, max_week + 1):
                zscore_dict = self.compute_zscore_scores(season, week)
                for name, zscore in zscore_dict.items():
                    zscore_rows.append({
                        'season': season,
                        'week': week,
                        'name': name,
                        'zscore': round(zscore, 4)
                    })
        
        df_zscore = pd.DataFrame(zscore_rows)
        df_zscore = df_zscore.sort_values(['season', 'week', 'zscore'], ascending=[True, True, False])
        df_zscore.to_csv(f'{output_dir}/zscore_scores.csv', index=False, encoding='utf-8-sig')
        print(f"导出Z-Score: {len(zscore_rows)} 条 -> {output_dir}/zscore_scores.csv")
        
        print(f"\n✓ 所有清洗后数据已导出到 {output_dir}/ 目录")
        return output_dir


def load_data(csv_path: str = '2026_MCM_Problem_C_Data.csv') -> DataProcessor:
    """便捷函数：加载数据"""
    return DataProcessor(csv_path)


if __name__ == '__main__':
    # 测试数据加载
    dp = DataProcessor('2026_MCM_Problem_C_Data.csv')
    
    print(f"总赛季数: {len(dp.get_seasons())}")
    print(f"赛季列表: {dp.get_seasons()}")
    
    # 测试第27季
    s27 = dp.get_contestants_in_season(27)
    print(f"\n第27季选手数: {len(s27)}")
    print(f"规则类型: {dp.get_rule_type(27)}")
    
    # 测试淘汰事件
    events = dp.get_elimination_events(27)
    print(f"淘汰事件数: {len(events)}")
    if events:
        print(f"第一个淘汰事件: 第{events[0].week}周淘汰 {events[0].eliminated}")
